### Introduction

A tool to anonymize images in ros2 bags. The tool combines GroundingDINO, OpenCLIP and SegmentAnything to anonymize images in rosbags

<p align="center">
    <img src="docs/rosbag2_anonymizer.png" alt="system" height="387px"/>
  <img src="docs/validation.png" alt="system" height="387px"/>
</p>



### Installation

**Clone the repository**

``` shell
git clone https://github.com/autowarefoundation/autoware_rosbag2_anonymizer.git
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

You should set your configuration in the configuration files under `config` folder according to the usage.
Following instructions will guide you to set each configuration file.

**`config/anonymize_with_unified_model.yaml`**

```yaml
rosbag:
  input_bag_path: "path/to/input.bag" # Path to the input ROS2 bag file with 'mcap' or 'sqlite3' extension
  output_bag_path: "path/to/output/folder" # Path to the output ROS2 bag folder
  output_save_compressed_image: True # Save images as compressed images (True or False)
  output_storage_id: "sqlite3" # Storage id for the output bag file (`sqlite3` or `mcap`)

grounding_dino:
  box_threshold: 0.1 # Threshold for the bounding box (float)
  text_threshold: 0.1 # Threshold for the text (float)
  nms_threshold: 0.1 # Threshold for the non-maximum suppression (float)

open_clip:
  score_threshold: 0.7 # Validity threshold for the OpenCLIP model (float

yolo:
  confidence: 0.15 # Confidence threshold for the YOLOv8 model (float)

bbox_validation:
  iou_threshold: 0.9 # Threshold for the intersection over union (float), if the intersection over union is greater than this threshold, the object will be selected as inside the validation prompt

blur:
  kernel_size: 31 # Kernel size for the Gaussian blur (int)
  sigma_x: 11 # Sigma x for the Gaussian blur (int)
```

**`config/yolo_create_dataset.yaml`**

```yaml
rosbag:
  input_bags_folder: "path/to/input/folder" # Path to the input ROS2 bag files folder

dataset:
  output_dataset_folder: "path/to/output/folder" # Path to the output dataset folder
  output_dataset_subsample_coefficient: 25 # Subsample coefficient for the dataset (int)

grounding_dino:
  box_threshold: 0.1 # Threshold for the bounding box (float)
  text_threshold: 0.1 # Threshold for the text (float)
  nms_threshold: 0.1 # Threshold for the non-maximum suppression (float)

open_clip:
  score_threshold: 0.7 # Validity threshold for the OpenCLIP model (float

bbox_validation:
  iou_threshold: 0.9 # Threshold for the intersection over union (float), if the intersection over union is greater than this threshold, the object will be selected as inside the validation prompt
```

**`config/yolo_train.yaml`**

```yaml
dataset:
  input_dataset_yaml: "path/to/input/data.yaml" # Path to the config file of the dataset, which is created in the previous step

yolo:
  epochs: 100 # Number of epochs for the YOLOv8 model (int)
  model: 'yolov8x.pt' # Select the base model for YOLOv8 ('yolov8x.pt' 'yolov8l.pt', 'yolov8m.pt', 'yolov8n.pt')
```

**`config/yolo_anonymize.yaml`**

```yaml
rosbag:
  input_bag_path: "path/to/input.bag" # Path to the input ROS2 bag file with 'mcap' or 'sqlite3' extension
  output_bag_path: "path/to/output/folder" # Path to the output ROS2 bag folder
  output_save_compressed_image: True # Save images as compressed images (True or False)
  output_storage_id: "sqlite3" # Storage id for the output bag file (`sqlite3` or `mcap`)

yolo:
  model: "path/to/yolo/model" # Path to the trained YOLOv8 model file (`.pt` extension) (you can download the pre-trained model from releases)
  config_path: "path/to/input/data.yaml" # Path to the config file of the dataset, which is created in the previous step
  confidence: 0.15 # Confidence threshold for the YOLOv8 model (float)

blur:
  kernel_size: 31 # Kernel size for the Gaussian blur (int)
  sigma_x: 11 # Sigma x for the Gaussian blur (int)
```

### Usage

The tool provides two options to anonymize images in rosbags.

**Option 1: Anonymize with Unified Model**

You should provide a single rosbag and tool anonymize images in rosbag with a unified model.
The model is a combination of GroundingDINO, OpenCLIP, YOLOv8 and SegmentAnything.
If you don't want to use pre-trained YOLOv8 model, you can follow the instructions in the second option to train your 
own YOLOv8 model.

You should set your configuration in `config/anonymize_with_unified_model.yaml` file.

``` shell
python3 main.py config/anonymize_with_unified_model.yaml --anonymize_with_unified_model
```

**Option 2: Anonymize Using the YOLOv8 Model Trained on a Dataset Created with the Unified Model**

<ins>Step 1:</ins> Create an initial dataset with the unified model.
You can provide multiple rosbags to create a dataset.
You should set your configuration in `config/yolo_create_dataset.yaml` file.
After running the following command, the tool will create a dataset in YOLO format.

``` shell
python3 main.py config/yolo_create_dataset.yaml --yolo_create_dataset
```

<ins>Step 2:</ins> The dataset which is created in the first step has some missing labels.
You should label the missing labels manually.

<ins>Step 3:</ins> After labeling the missing labels, you should split the dataset into train and validation sets.
If the labeling tool you used does not provide a split option, you can use the following command to split the dataset.

Give the path to the dataset folder which is created in the first step.

``` shell
autoware-rosbag2-anonymizer-split-dataset /path/to/dataset
```

<ins>Step 4:</ins> Train the YOLOv8 model with the dataset.
You should set your configuration in `config/yolo_train.yaml` file.

``` shell
python3 main.py config/yolo_train.yaml --yolo_train
```

<ins>Step 5:</ins> Anonymize images in rosbags with the trained YOLOv8 model.
You should set your configuration in `config/yolo_anonymize.yaml` file. 
If you want to anonymize your ROS2 bag file with only YOLOv8 model,
you should use following command.
But we recommend to use the unified model for better results.
You can follow the `Option 1` for the unified model with the YOLOv8 model trained by you.

``` shell
python3 main.py config/yolo_anonymize.yaml --yolo_anonymize
```