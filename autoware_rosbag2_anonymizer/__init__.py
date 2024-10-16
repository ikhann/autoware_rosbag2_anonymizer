import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

from autoware_rosbag2_anonymizer.common import create_classes, calculate_iou, bbox_check, blur_detections

from autoware_rosbag2_anonymizer.rosbag_io.rosbag_reader import RosbagReader
from autoware_rosbag2_anonymizer.rosbag_io.rosbag_writer import RosbagWriter

from autoware_rosbag2_anonymizer.model.open_clip import OpenClipModel
from autoware_rosbag2_anonymizer.model.grounding_dino import GroundingDINO
from autoware_rosbag2_anonymizer.model.sam import SAM
from autoware_rosbag2_anonymizer.model.sam2 import SAM2
from autoware_rosbag2_anonymizer.model.yolo import Yolo
from autoware_rosbag2_anonymizer.model.unified_language_model import UnifiedLanguageModel