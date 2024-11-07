from autoware_rosbag2_anonymizer.model.unified_language_model import (
    UnifiedLanguageModel,
)

from collections import Counter
from tqdm import tqdm

import os
import yaml
import json

import torch
import cv2


class Validator:
    def __init__(self, config_data, json_data, device):
        self.device = device
        self.unified_language_model = UnifiedLanguageModel(
            config_data, json_data, device
        )

        self.root_dir = config_data["dataset"]["input_dataset_yaml"]
        if not self.root_dir.split(".")[-1] in {"yaml", "yml"}:  # bool: is yaml ot Not
            print("Root directory is not valid. Please provide a valid root directory.")
            exit(1)
        self.labels, self.number_of_classes = self.read_dataset_yaml()
        self.image_dir = "/".join(self.root_dir.split("/")[:-1]) + "/valid/images"
        self.label_dir = "/".join(self.root_dir.split("/")[:-1]) + "/valid/labels"

        self.max_samples = config_data["dataset"]["max_samples"]

        print("Image directory:", self.image_dir)
        print("Label directory:", self.label_dir)
        print("Labels:", self.labels)
        print("Number of classes:", self.number_of_classes)

    @staticmethod
    def convert_xywh_to_xyxy(normalized_box, image_width, image_height):
        """
        Convert a bounding box from normalized xywh format to pixel xyxy format.

        Parameters:
            normalized_box: list or tuple (class_id, x_center, y_center, width, height) with values normalized from 0 to 1
            image_width: int, width of the image in pixels
            image_height: int, height of the image in pixels

        Returns:
            list: [x_min, y_min, x_max, y_max] in pixel coordinates
        """
        class_id, x_center, y_center, width, height = normalized_box

        # Convert to pixel coordinates
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)

        return [class_id, x_min, y_min, x_max, y_max]

    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
            box1: list, tuple or array [x1, y1, x2, y2]
            box2: list, tuple or array [x1, y1, x2, y2]

        Returns:
            float: IoU between box1 and box2
        """
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        # Compute the area of intersection
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Compute the IoU
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    @staticmethod
    def find_best_ground_truth_match(pred_box, ground_truth_boxes, iou_threshold=0.5):
        """
        Find the ground-truth box with the highest IoU for a single predicted box.

        Parameters:
            pred_box: list or array [x1, y1, x2, y2] representing a single predicted bounding box.
            ground_truth_boxes: list of lists or array of bounding boxes for ground truths (e.g., [[class_id, x1, y1, x2, y2], ...])
            iou_threshold: float, IoU threshold to consider a match

        Returns:
            tuple: (best_ground_truth_box, best_iou) if a match is found; otherwise (None, 0)
        """
        best_iou = 0
        best_gt = None

        for gt_box in ground_truth_boxes:
            iou = Validator.calculate_iou(pred_box, gt_box[1:])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_box

        # Return the best match if IoU is above the threshold, otherwise None
        if best_iou >= iou_threshold:
            return best_gt, best_iou
        else:
            return None, 0

    def read_dataset_yaml(self):
        """
        Read the dataset YAML file and return the contents.

        YAML file should have YOLO format.

        Returns:
            list: labels
            int: number of classes
        """
        with open(self.root_dir, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data["names"], yaml_data["nc"]

    def calculate_precision_recall(
        self, true_positives, false_positives, false_negatives
    ):
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        return precision, recall

    def validate_dataset(self):
        """
        Validate the dataset using the Unified Model.
        """

        true_positives = [0 for _ in range(self.number_of_classes)]
        false_positives = [0 for _ in range(self.number_of_classes)]
        false_negatives = [0 for _ in range(self.number_of_classes)]

        counter = 0
        label_files = os.listdir(self.label_dir)
        for label_file in tqdm(
            label_files,
            "Validating Dataset",
            total=len(label_files) if self.max_samples == -1 else self.max_samples,
        ):
            # Load the image and predict the bounding boxes
            image = cv2.imread(
                os.path.join(self.image_dir, label_file.split(".")[0] + ".jpg")
            )
            detections = self.unified_language_model(image)

            # Get GT boxes for the image
            ground_truths = []
            with open(os.path.join(self.label_dir, label_file), "r") as file:
                lines = file.readlines()
                for line in lines:
                    class_id, x_center, y_center, width, height = map(
                        float, line.split(" ")
                    )
                    ground_truths.append(
                        self.convert_xywh_to_xyxy(
                            (int(class_id), x_center, y_center, width, height),
                            image.shape[1],
                            image.shape[0],
                        )
                    )

            # Match the predicted boxes with the ground truth boxes
            matched_count = [0 for _ in range(self.number_of_classes)]
            for xyxy, _, confidence, class_id, _, _ in detections:
                x_min, y_min, x_max, y_max = xyxy
                pred_box = [x_min, y_min, x_max, y_max]

                best_gt, best_iou = self.find_best_ground_truth_match(
                    pred_box, ground_truths
                )

                if best_gt is not None:
                    matched_count[best_gt[0]] += 1
                    true_positives[best_gt[0]] += 1
                else:
                    false_positives[class_id] += 1

            # Calculate false negatives
            ground_truths_per_class = Counter(gt_box[0] for gt_box in ground_truths)
            false_negatives = [
                fn + max(0, ground_truths_per_class.get(i, 0) - match)
                for i, (fn, match) in enumerate(zip(false_negatives, matched_count))
            ]

            counter += 1
            if counter >= self.max_samples & self.max_samples != -1:
                break

        print("True Positives:", true_positives)
        print("False Positives:", false_positives)
        print("False Negatives:", false_negatives)

        precision_list = []
        recall_list = []
        AP_per_class = []
        for cls_name, tp, fp, fn in zip(
            self.labels, true_positives, false_positives, false_negatives
        ):
            precision, recall = self.calculate_precision_recall(tp, fp, fn)
            AP = precision  # Assuming single threshold AP at IoU=0.5
            precision_list.append(precision)
            recall_list.append(recall)
            AP_per_class.append(AP)

            # Print precision, recall, and AP for each class
            print(f"Class: {cls_name}")
            print(f"  Precision: {precision:.6f}")
            print(f"  Recall: {recall:.6f}")
            print(f"  AP@0.5: {AP:.6f}\n")

        # Calculate overall precision, recall, and mAP across all classes
        total_tp = sum(true_positives)
        total_fp = sum(false_positives)
        total_fn = sum(false_negatives)

        # Overall metrics
        overall_precision, overall_recall = self.calculate_precision_recall(
            total_tp, total_fp, total_fn
        )
        overall_mAP = sum(AP_per_class) / len(AP_per_class)  # Average AP across classes

        # Print overall metrics
        print("Overall Metrics:")
        print(f"  Overall Precision: {overall_precision:.6f}")
        print(f"  Overall Recall: {overall_recall:.6f}")
        print(f"  Overall mAP@0.5: {overall_mAP:.6f}")
        
        with open("validation_results.txt", "w") as file:
            file.write(f"Validation Results of: {self.root_dir}\n")
            file.write("------------------------------\n")
            file.write("Labels: " + str(self.labels) + "\n")
            file.write("True Positives: " + str(true_positives) + "\n")
            file.write("False Positives: " + str(false_positives) + "\n")
            file.write("False Negatives: " + str(false_negatives) + "\n")
            file.write("Precision: " + str(precision_list) + "\n")
            file.write("Recall: " + str(recall_list) + "\n")
            file.write("mAP: " + str(overall_mAP) + "\n")


if __name__ == "__main__":
    # Load configuration and JSON data
    with open(
        "config/validation.yaml",
        "r",
    ) as file:
        config_data = yaml.safe_load(file)

    with open("validation.json", "r") as json_file:
        json_data = json.load(json_file)

    # Initialize the validator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validator = Validator(config_data, json_data, device)

    # Validate the dataset
    validator.validate_dataset()
