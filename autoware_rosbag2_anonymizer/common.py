from typing import List, Tuple, Dict

import cv2

import os
import yaml


def get_file_paths(root_folder: str, extensions: List[str]):
    file_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    file_paths.append(os.path.join(root, file))
                    break
    return file_paths


def create_classes(json_data) -> Tuple[List[str], List[str], Dict]:
    detection_classes = []
    prompts = []
    prompts_map = {}

    for data in json_data["prompts"]:
        if data["prompt"] not in detection_classes:
            detection_classes.append(data["prompt"])

    for data in json_data["prompts"]:
        if data["prompt"] not in prompts:
            prompts.append(data["prompt"])
        [
            prompts.append(prompt)
            for prompt in data["should_inside"]
            if prompt not in prompts
        ]
        [
            prompts.append(prompt)
            for prompt in data["should_not_inside"]
            if prompt not in prompts
        ]

    for data in json_data["prompts"]:
        prompts_map[data["prompt"]] = {
            "should_inside": data["should_inside"],
            "should_not_inside": data["should_not_inside"],
        }

    return detection_classes, prompts, prompts_map


def create_yolo_classes(yolo_config_path) -> List[str]:
    with open(yolo_config_path, "r") as file:
        yolo_config = yaml.safe_load(file)

    return yolo_config["names"]


def calculate_iou(box1, box2) -> int:
    A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    if x2_i > x1_i and y2_i > y1_i:
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        iou = inter_area / min(A1, A2)
        return iou
    else:
        return 0.0


def bbox_check(xyxy, class_id, detections, iou_threshold, classes, class_map) -> bool:
    for valid_xyxy, _, _, valid_class_id, _, _ in detections:
        if classes[valid_class_id] in class_map[classes[class_id]]["should_inside"]:
            iou = calculate_iou(xyxy, valid_xyxy)
            if iou >= iou_threshold:
                return True
    return False


def blur_detections(img, detections, kernel_size, sigma_x):
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_x)
    output_img = output = img.copy()
    for _, mask, _, _, _, _ in detections:
        output[mask] = blurred_img[mask]
    return output_img
