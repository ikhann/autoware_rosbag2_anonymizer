import torch
import torchvision

import cv2
from PIL import Image

import supervision as sv

import numpy as np

from autoware_rosbag2_anonymizer.model.open_clip import OpenClipModel
from autoware_rosbag2_anonymizer.model.grounding_dino import GroundingDINO
from autoware_rosbag2_anonymizer.model.sam import SAM

from autoware_rosbag2_anonymizer.common import (
    create_classes,
    bbox_check,
)


class UnifiedLanguageModel:
    def __init__(self, config, json) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detection_classes, self.classes, self.class_map = create_classes(
            json_data=json
        )

        self.grounding_dino_config_path = config["grounding_dino"]["config_path"]
        self.grounding_dino_checkpoint_path = config["grounding_dino"][
            "checkpoint_path"
        ]

        self.box_threshold = config["grounding_dino"]["box_threshold"]
        self.text_threshold = config["grounding_dino"]["text_threshold"]
        self.nms_threshold = config["grounding_dino"]["nms_threshold"]

        self.sam_encoder_version = config["segment_anything"]["encoder_version"]
        self.sam_checkpoint_path = config["segment_anything"]["checkpoint_path"]

        self.openclip_model_name = config["openclip"]["model_name"]
        self.openclip_pretrained_model = config["openclip"]["pretrained_model"]
        self.openclip_score_threshold = config["openclip"]["score_threshold"]

        self.iou_threshold = config["bbox_validation"]["iou_threshold"]

        # Grounding DINO
        self.grounding_dino = GroundingDINO(
            self.grounding_dino_config_path, self.grounding_dino_checkpoint_path
        )

        # Segment-Anything
        self.sam = SAM(self.sam_encoder_version, self.sam_checkpoint_path, self.device)

        # Openclip
        self.open_clip = OpenClipModel(
            self.openclip_model_name, self.openclip_pretrained_model
        )

    def __call__(self, image: cv2.Mat) -> sv.Detections:
        # Run DINO
        detections = self.grounding_dino(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # Remove class_id if it is 'None'
        not_nons = [
            index
            for index, (_, _, _, class_id, _, _) in enumerate(detections)
            if class_id is not None
        ]
        detections.xyxy = detections.xyxy[not_nons]
        detections.confidence = detections.confidence[not_nons]
        detections.class_id = detections.class_id[not_nons]

        # NMS
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.nms_threshold,
            )
            .numpy()
            .tolist()
        )
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # Validation
        valid_ids = []
        invalid_ids = []
        for index, (xyxy, _, _, class_id, _, _) in enumerate(detections):
            if self.classes[class_id] in self.detection_classes:
                # Run OpenClip
                # and accept as a valid object if the score is greater than 0.9
                detection_image = image[
                    int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2]), :
                ]
                pil_image = Image.fromarray(detection_image)
                scores = self.open_clip(pil_image, self.classes)
                if scores.numpy().tolist()[0][class_id] > self.openclip_score_threshold:
                    valid_ids.append(index)
                    continue

                # Bbox validation
                # If the object is within the 'should_inside' object
                # and if the score is the highest among the scores,
                # or greater than 0.4.
                if bbox_check(
                    xyxy,
                    class_id,
                    detections,
                    self.iou_threshold,
                    self.classes,
                    self.class_map,
                ) and (
                    max(scores.numpy().tolist()[0])
                    == scores.numpy().tolist()[0][class_id]
                    or scores.numpy().tolist()[0][class_id] > 0.3
                ):
                    valid_ids.append(index)
                else:
                    invalid_ids.append(index)
            else:
                invalid_ids.append(index)
        detections.xyxy = detections.xyxy[valid_ids]
        detections.confidence = detections.confidence[valid_ids]
        detections.class_id = np.array(
            [
                self.detection_classes.index(self.classes[x])
                for x in detections.class_id[valid_ids]
            ]
        )

        return detections
