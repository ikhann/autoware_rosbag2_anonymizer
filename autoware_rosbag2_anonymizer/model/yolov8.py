from ultralytics import YOLO

import supervision as sv


class Yolov8:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def __call__(self, image) -> sv.Detections:
        results = self.model(image)

        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )
        return detections

    def train(self, data_yaml: str, epochs: int):
        self.model.train(data=data_yaml, epochs=epochs)
