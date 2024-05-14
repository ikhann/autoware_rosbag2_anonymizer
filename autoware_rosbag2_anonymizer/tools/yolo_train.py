from autoware_rosbag2_anonymizer.model.yolov8 import Yolov8

def yolo_train(config_data) -> None:
    yolo_model = Yolov8(config_data["yolo"]["model"])
    yolo_model.train(
        data_yaml=config_data["dataset"]["input_dataset_yaml"],
        epochs=config_data["yolo"]["epochs"]
    )