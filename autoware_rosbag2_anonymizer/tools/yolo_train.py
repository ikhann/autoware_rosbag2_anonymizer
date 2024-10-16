from autoware_rosbag2_anonymizer.model.yolo import Yolo

def yolo_train(config_data) -> None:
    yolo_model = Yolo(config_data["yolo"]["model"])
    yolo_model.train(
        data_yaml=config_data["dataset"]["input_dataset_yaml"],
        epochs=config_data["yolo"]["epochs"]
    )