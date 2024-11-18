import argparse
import yaml
import json

import torch

from autoware_rosbag2_anonymizer.tools.anonymize_with_unified_model import (
    anonymize_with_unified_model,
)
from autoware_rosbag2_anonymizer.tools.yolo_create_dataset import (
    yolo_create_dataset,
)
from autoware_rosbag2_anonymizer.tools.yolo_train import (
    yolo_train,
)
from autoware_rosbag2_anonymizer.tools.yolo_anonymize import (
    yolo_anonymize,
)
from autoware_rosbag2_anonymizer.tools.validator import (
    Validator,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="ROS2 image anonymizer tool.")

    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument(
        "--anonymize_with_unified_model",
        action="store_true",
        help="Run a ROS 2 node to anonymize images received from a subscription topic using Grounding DINO, OpenCLIP, and SAM.",
    )
    parser.add_argument(
        "--yolo_create_dataset",
        action="store_true",
        help="Create initial dataset in YOLO format by combining Grounding DINO, OpenCLIP, and SAM.",
    )
    parser.add_argument(
        "--yolo_train",
        action="store_true",
        help="Train YOLO with the created dataset.",
    )
    parser.add_argument(
        "--yolo_anonymize",
        action="store_true",
        help="Run YOLO to anonymize images received from a subscription topic.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Validate the dataset using the Unified Model.",
    )

    args = parser.parse_args()

    if not any(
        [
            args.anonymize_with_unified_model,
            args.yolo_create_dataset,
            args.yolo_train,
            args.yolo_anonymize,
            args.validation,
        ]
    ):
        parser.error(
            "Please select one of --anonymize_with_unified_model, --yolo_create_dataset, --yolo_train, --yolo_anonymize, or --validation."
        )

    return args


if __name__ == "__main__":
    args = parse_arguments()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration file
    with open(args.config, "r") as file:
        config_data = yaml.safe_load(file)

    # Load detection classes or model-specific metadata
    with open("validation.json", "r") as json_file:
        json_data = json.load(json_file)

    # Determine the selected action and execute the corresponding function
    if args.anonymize_with_unified_model:
        anonymize_with_unified_model(
            config_data=config_data,
            json_data=json_data,
            device=DEVICE,
        )
    elif args.yolo_create_dataset:
        yolo_create_dataset(
            config_data=config_data,
            json_data=json_data,
            device=DEVICE,
        )
    elif args.yolo_train:
        yolo_train(
            config_data=config_data,
        )
    elif args.yolo_anonymize:
        yolo_anonymize(
            config_data=config_data,
            json_data=json_data,
            device=DEVICE,
        )
    elif args.validation:
        validator = Validator(
            config_data,
            json_data,
            DEVICE,
        )
        validator.validate_dataset()
