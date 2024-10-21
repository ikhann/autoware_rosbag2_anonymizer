from groundingdino.util.inference import Model
from groundingdino.util.inference import load_model, predict

import numpy as np

import torch
import torchvision.transforms.functional as F

import supervision as sv
import cv2


class GroundingDINO:
    def __init__(self, config_path, checkpoint_path, device) -> None:
        self.model = self.load_model(config_path, checkpoint_path, device)
        self.device = device

    def load_model(self, config_path, checkpoint_path, device):
        grounding_model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=device,
        )
        return grounding_model

    # def preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
    #     transform = T.Compose(
    #         [
    #             T.RandomResize([800], max_size=1333),
    #             T.ToTensor(),
    #             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #         ]
    #     )

    #     image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    #     image_transformed, _ = transform(image_pillow, None)

    #     return image_transformed

    def preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_tensor = (
            torch.from_numpy(image_rgb).float().permute(2, 0, 1).to("cuda") / 255.0
        )  # Remove unsqueeze

        h, w = image_tensor.shape[1:3]
        scale_factor = 800 / min(h, w)
        max_size = 1333
        if max(h, w) * scale_factor > max_size:
            scale_factor = max_size / max(h, w)

        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        image_tensor = F.resize(image_tensor, [new_h, new_w])

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(
            3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(
            3, 1, 1
        )
        image_tensor = (image_tensor - mean) / std

        return image_tensor

    def __call__(self, image, classes, box_threshold, text_threshold) -> sv.Detections:
        boxes, confidences, labels = predict(
            model=self.model,
            image=self.preprocess_image(image).to(self.device),
            caption=". ".join(classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h, source_w=source_w, boxes=boxes, logits=confidences
        )
        class_id = Model.phrases2classes(phrases=labels, classes=classes)
        detections.class_id = class_id

        return detections
