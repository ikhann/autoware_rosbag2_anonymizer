import cv2
import cv_bridge

import supervision as sv

from autoware_rosbag2_anonymizer.common import (
    create_yolo_classes,
    blur_detections,
)

from autoware_rosbag2_anonymizer.model.yolov8 import Yolov8
from autoware_rosbag2_anonymizer.model.sam import SAM

from autoware_rosbag2_anonymizer.rosbag_io.rosbag_reader import RosbagReader
from autoware_rosbag2_anonymizer.rosbag_io.rosbag_writer import RosbagWriter


def yolo_anonymize(config_data, json_data, device) -> None:
    
    yolo_model_path = config_data["yolo"]["model"]
    yolo_confidence = config_data["yolo"]["confidence"]
    yolo_config_path = config_data["yolo"]["config_path"]

    # YOLOv8
    yolo_model = Yolov8(yolo_model_path)

    # Declare classes for YOLOv8 from yaml file
    CLASSES = create_yolo_classes(yolo_config_path)

    # Segment-Anything
    SAM_ENCODER_VERSION = config_data["segment_anything"]["encoder_version"]
    SAM_CHECKPOINT_PATH = config_data["segment_anything"]["checkpoint_path"]
    sam = SAM(SAM_ENCODER_VERSION, SAM_CHECKPOINT_PATH, device)

    # Create rosbag reader and rosbag writer
    reader = RosbagReader(config_data["rosbag"]["input_bag_path"], 1)
    writer = RosbagWriter(
        config_data["rosbag"]["output_bag_paht"],
        config_data["rosbag"]["output_save_compressed_image"],
        config_data["rosbag"]["output_storage_id"],
    )

    for i, (msg, is_image) in enumerate(reader):
        if not is_image:
            writer.write_any(msg.data, msg.type, msg.topic, msg.timestamp)
        else:
            image = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(msg.data)

            detections = yolo_model(image, confidence=yolo_confidence)

            # Run SAM
            detections = sam(image=image, detections=detections)

            # Blur detections
            output = blur_detections(
                image,
                detections,
                config_data["blur"]["kernel_size"],
                config_data["blur"]["sigma_x"],
            )
            
            # Debug ------------------
            # bounding_box_annotator = sv.BoundingBoxAnnotator()
            # annotated_image = bounding_box_annotator.annotate(
            #     scene=output,
            #     detections=detections,
            # )

            # labels = [
            #     f"{CLASSES[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _, _ in detections
            # ]
            # label_annotator = sv.LabelAnnotator()
            # annotated_image = label_annotator.annotate(
            #     output,
            #     detections,
            #     labels,
            # )

            # height, width = image.shape[:2]
            # annotated_image = cv2.resize(annotated_image, (width // 2, height // 2))
            # cv2.imshow("test", annotated_image)
            # cv2.waitKey(1)
            # Debug ------------------
