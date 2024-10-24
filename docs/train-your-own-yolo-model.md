## Train Your Own YOLO Model

This document explains how to train your own YOLO model and anonymize images in ROS2 bag files with the trained YOLO model or the unified model.

### 1. Create An Initial Dataset

Create an initial dataset with the unified model. You can provide multiple rosbags to create a dataset. You should set your configuration in `config/yolo_create_dataset.yaml` file. After running the following command, the tool will create a dataset in YOLO format.

``` shell
python3 main.py config/yolo_create_dataset.yaml --yolo_create_dataset
```

### 2. Label Missing Labels

The dataset which is created in the first step has some missing labels.
You should label the missing labels manually. You can use the following example tools to label the missing labels:

- [label-studio](https://github.com/HumanSignal/label-studio)
- [Roboflow](https://roboflow.com/) (You can use the free version)

### 3. Create Train and Validation Sets

After labeling the missing labels, you should split the dataset into train and validation sets.
If the labeling tool you used does not provide a split option, you can use the following command to split the dataset.

Give the path to the dataset folder which is created in the first step.

``` shell
autoware-rosbag2-anonymizer-split-dataset /path/to/dataset
```

### 4. Train the YOLO Model

Train the YOLO model with the dataset.
You should set your configuration in `config/yolo_train.yaml` file.

``` shell
python3 main.py config/yolo_train.yaml --yolo_train
```

### 5. Anonymize Images in ROS2 Bag Files

Anonymize images in ROS2 bag files with the trained YOLO model.
You should set your configuration in `config/yolo_anonymize.yaml` file. 
If you want to anonymize your ROS2 bag file with only YOLO model,
you should use following command.
But we recommend to use the unified model for better results.
You can follow the `Option 1` for the unified model with the YOLO model trained by you.

``` shell
python3 main.py config/yolo_anonymize.yaml --yolo_anonymize
```