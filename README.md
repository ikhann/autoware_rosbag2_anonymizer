### Introduction

A tool to anonymize images in ros2 bags. The tool combines GroundingDINO, OpenCLIP and SegmentAnything to anonymize images in rosbags

<p align="center">
    <img src="docs/rosbag2_anonymizer.png" alt="system" height="387px"/>
  <img src="docs/validation.png" alt="system" height="387px"/>
</p>



### Installation

**Clone the repository**

``` shell
git clone git@github.com:leo-drive/rosbag2_anonymizer.git
cd rosbag2_anonymizer
```

**Download the pretrained weights**

``` shell
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
```

**Install ros2 mcap dependencies if you will use mcap files**

``` shell
sudo apt install ros-humble-rosbag2-storage-mcap
```

**Install `autoware_rosbag2_anonymizer` package**


``` shell
python3 -m pip install .
```

### Configuration

Define prompts in the `validation.json`file. The tool will use these prompts to detect objects.
You can add your prompts as dictionaries under the `prompts` key. Each dictionary should have two keys:
- `prompt`: The prompt that will be used to detect the object. This prompt will be blurred in the anonymization process.
- `should_inside`: This is a list of prompts that object should be inside. If the object is not inside the prompts, the tool will not blur the object.

``` json
{
    "prompts": [
        {
            "prompt": "license plate",
            "should_inside": ["car", "bus", "..."]
        },
        {
            "prompt": "human face",
            "should_inside": ["person", "human body", "..."]
        }
    ]
}
```

Also, you should set your configuration in the configuration files under `config` folder according to the usage.

### Usage

The tool provides two options to anonymize images in rosbags.

**Option 1: Anonymize with Unified Model**

You should provide a single rosbag and tool anonymize images in rosbag with a unified model.
The model is a combination of GroundingDINO, OpenCLIP and SegmentAnything.

You should set your configuration in `config/anonymize_with_unified_model.yaml` file.

``` shell
python3 main.py config/anonymize_with_unified_model.yaml --anonymize_with_unified_model
```

**Option 2: Anonymize Using the YOLOv8 Model Trained on a Dataset Created with the Unified Model**

<ins>Step 1:</ins> Create an initial dataset with the unified model.
You can provide multiple rosbags to create a dataset.
You should set your configuration in `config/create_dataset_with_unified_model.yaml` file.
After running the following command, the tool will create a dataset in YOLO format.

``` shell
python3 main.py config/yolo_create_dataset.yaml --yolo_create_dataset
```

<ins>Step 2:</ins> The dataset which is created in the first step has some missing labels.
You should label the missing labels manually.

<ins>Step 3:</ins> After labeling the missing labels, you should split the dataset into train and validation sets.
If the labeling tool you used does not provide a split option, you can use the following command to split the dataset.

``` shell
TODO
```

<ins>Step 4:</ins> Train the YOLOv8 model with the dataset.
You should set your configuration in `config/yolo_train.yaml` file.

``` shell
python3 main.py config/yolo_train.yaml --yolo_train
```

<ins>Step 5:</ins> Anonymize images in rosbags with the trained YOLOv8 model.
You should set your configuration in `config/yolo_anonymize.yaml` file.

``` shell
python3 main.py config/yolo_anonymize.yaml --yolo_anonymize
```