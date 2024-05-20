import cv_bridge

import supervision as sv
import cv2

from autoware_rosbag2_anonymizer.rosbag_io.rosbag_reader import RosbagReader
from autoware_rosbag2_anonymizer.rosbag_io.rosbag_writer import RosbagWriter

from autoware_rosbag2_anonymizer.model.unified_language_model import (
    UnifiedLanguageModel,
)
from autoware_rosbag2_anonymizer.model.sam import SAM

from autoware_rosbag2_anonymizer.common import (
    create_classes,
    blur_detections,
)


def anonymize_with_unified_model(config_data, json_data, device) -> None:
    reader = RosbagReader(config_data["rosbag"]["input_bag_path"], 1)
    writer = RosbagWriter(
        config_data["rosbag"]["output_bag_path"],
        config_data["rosbag"]["output_save_compressed_image"],
        config_data["rosbag"]["output_storage_id"],
    )

    # Segment-Anything parameters
    SAM_ENCODER_VERSION = config_data["segment_anything"]["encoder_version"]
    SAM_CHECKPOINT_PATH = config_data["segment_anything"]["checkpoint_path"]

    # Segment-Anything
    sam = SAM(SAM_ENCODER_VERSION, SAM_CHECKPOINT_PATH, device)

    unified_language_model = UnifiedLanguageModel(config_data, json_data)

    for i, (msg, is_image) in enumerate(reader):
        if not is_image:
            writer.write_any(msg.data, msg.type, msg.topic, msg.timestamp)
        else:
            # Convert image msg to cv.Mat
            image = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(msg.data)

            # Find bounding boxes with Unified Model
            detections = unified_language_model(image)

            # Run SAM
            detections = sam(image=image, detections=detections)

            # Blur detections
            output = blur_detections(
                image,
                detections,
                config_data["blur"]["kernel_size"],
                config_data["blur"]["sigma_x"],
            )

            # Write blured image to rosbag
            writer.write_image(output, msg.topic, msg.timestamp)

            # Debug ------------------
            # DETECTION_CLASSES, CLASSES, CLASS_MAP = create_classes(json_data=json_data)

            # bounding_box_annotator = sv.BoundingBoxAnnotator()
            # annotated_image = bounding_box_annotator.annotate(
            #     scene=output,
            #     detections=detections,
            # )

            # labels = [
            #     f"{DETECTION_CLASSES[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _, _ in detections
            # ]
            # label_annotator = sv.LabelAnnotator()
            # annotated_image = label_annotator.annotate(
            #     output,
            #     detections,
            #     labels,
            # )

            # height, width = image.shape[:2]
            # annotated_image = cv2.resize(annotated_image, (width // 2, height // 2))
            # cv2.imshow("test", annotated_image)
            # cv2.waitKey(1)
            # Debug ------------------
