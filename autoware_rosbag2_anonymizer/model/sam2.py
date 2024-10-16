from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import supervision as sv
import numpy as np


class SAM2:
    def __init__(self, model_cfg, checkpoint_path, device) -> None:
        self.sam2_predictor = self.load_sam2_predictor(
            model_cfg, checkpoint_path, device
        )

    def load_sam2_predictor(
        self, model_cfg, checkpoint_path, device
    ) -> SAM2ImagePredictor:
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        return sam2_predictor

    def __call__(self, image, detections: sv.Detections) -> sv.Detections:
        if detections.xyxy.size == 0: return detections

        self.sam2_predictor.set_image(image)
        masks, scores, logits = self.sam2_predictor.predict(
            box=detections.xyxy, multimask_output=False
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        detections.mask = masks.astype(bool)
        return detections
