## Validation

This document shows validation results of unified model and explains how to validate the anonymized images in ROS2 bag files with the unified model.

### What metrics are used for validation?

We use the following metrics for validation:

1. **Recall**: Recall is the ratio of the number of true positive predictions to the number of ground truth objects. Recall is calculated as follows:

    Recall = TP / (TP + FN)

    where:
    - TP: True Positive
    - FN: False Negative

2. **Precision**: Precision is the ratio of the number of true positive predictions to the number of predicted objects. Precision is calculated as follows:
   
    Precision = TP / (TP + FP)

    where:
    - TP: True Positive
    - FP: False Positive

3. **mAP (mean Average Precision)**: mAP is the average of the AP (Average Precision) of all classes. AP is the area under the precision-recall curve. We use the mAP@0.5 metric, which is the mAP calculated with IoU (Intersection over Union) threshold of 0.5.

### Results of Validation

We validated the unified model using a custom dataset containing approximately 1,500 images in YOLO format. This dataset includes two classes: human face and license plate.

The following table shows the validation results of the unified model:

| Class        | Recall | Precision | mAP@0.5 |
|--------------|--------|-----------|---------|
| Human Face   | 0.9524 | 0.5512    | 0.5512  |
| License Plate| 0.9349 | 0.7018    | 0.7018  |
| **Average**  | 0.9362 | 0.6879    | 0.6265  |

- When we examined the validation results, we observed that the unified model performed well in detecting human faces and license plates. The model achieved a high recall rate for both classes. However, the precision rate was lower than the recall rate for both classes. This is because the model detects a lot of false positives. Since we using model to anonymize objects in images, we can tolerate false positives to some extent.

### How Can You Validate Unified Model with Your Own Dataset?

Firstly you should have a dataset in YOLO format:

``` shell
example_dataset
├── train
│   ├── images
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── labels
│       ├── 0001.txt
│       ├── 0002.txt
│       └── ...
└── valid
│   ├── images
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── labels
│       ├── 0001.txt
│       ├── 0002.txt
│       └── ...
└── data.yaml
```

Then you can validate the unified model with the following command:

``` shell
python3 main.py config/validation.yaml --validation
```

You should set your configuration in `config/validation.yaml` file. You can set the following parameters in the configuration file:

```yaml
dataset:
  input_dataset_yaml: "path/to/data.yaml" # Path to the config file of the dataset, which is created in the previous step
  max_samples: -1 # Maximum number of samples to use for validation (int), if -1, all samples will be used

  grounding_dino:
  box_threshold: 0.1 # Threshold for the bounding box (float)
  text_threshold: 0.1 # Threshold for the text (float)
  nms_threshold: 0.1 # Threshold for the non-maximum suppression (float)

open_clip:
  score_threshold: 0.7 # Validity threshold for the OpenCLIP model (float

bbox_validation:
  iou_threshold: 0.9 # Threshold for the intersection over union (float), if the intersection over union is greater than this threshold, the object will be selected as inside the validation prompt
```