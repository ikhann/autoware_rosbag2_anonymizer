import os
import cv2
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import math
import supervision as sv

from autoware_rosbag2_anonymizer.model.unified_language_model import UnifiedLanguageModel
from autoware_rosbag2_anonymizer.model.sam2 import SAM2
from autoware_rosbag2_anonymizer.common import create_classes


class ImageAnonymizerNode(Node):
    def __init__(self, config_data, json_data, device):
        super().__init__("image_anonymizer_node")
        self.bridge = CvBridge()
        self.config_data = config_data
        self.json_data = json_data
        self.device = device

        # Initialize models
        self.sam2 = SAM2(
            config_data["segment_anything_2"]["model_cfg"],
            config_data["segment_anything_2"]["checkpoint_path"],
            device,
        )
        self.unified_language_model = UnifiedLanguageModel(config_data, json_data, device)

        # Subscription to the input image topic
        self.subscription = self.create_subscription(
            Image,
            config_data["ros"]["input_image_topic"],
            self.image_callback,
            1
        )

        # Subscription to the input camera info topic
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            config_data["ros"]["input_camera_info_topic"],
            self.camera_info_callback,
            1
        )

        self.declare_parameter("prompts", ["face", "box"])
        self.add_on_set_parameters_callback(self.on_prompts_update)

        # Camera intrinsic parameters
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Create classes for annotations
        self.DETECTION_CLASSES, self.CLASSES, self.CLASS_MAP = create_classes(json_data)


    def on_prompts_update(self, params):
        for param in params:
            if param.name == "prompts" and isinstance(param.value, list):
                new_prompts = param.value
                self.get_logger().info(f"Updating prompts: {new_prompts}")

                # Ensure json_data contains all prompts with empty fields for 'should_inside' and 'should_not_inside'
                self.json_data["prompts"] = [
                    {"prompt": prompt, "should_inside": [], "should_not_inside": []}
                    for prompt in new_prompts
                ]

                # Update the UnifiedLanguageModel with the new json_data
                self.unified_language_model = UnifiedLanguageModel(
                    self.config_data, self.json_data, self.device
                )

                self.DETECTION_CLASSES, self.CLASSES, self.CLASS_MAP = create_classes(self.json_data)

                self.get_logger().info(
                    f"Prompts and models updated successfully: {self.json_data['prompts']}")

        return SetParametersResult(successful=True)


    def image_callback(self, msg):
        if self.fx is None:
            return

        self.get_logger().info("Received an image message")
        # Convert ROS Image message to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Perform anonymization
        anonymized_image, detections = self.anonymize_image(image)

        # Log detections
        for i, (bbox, confidence, class_id) in enumerate(
                zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            self.get_logger().info(
                f"Detection {i}: Class={self.DETECTION_CLASSES[class_id]}, "
                f"Confidence={confidence:.2f}, Width={width}, Height={height}"
            )

        # Print debug information
        if self.config_data["debug"]["print_on_terminal"]:
            self.print_detections(detections)

        # Display the processed image
        if self.config_data["debug"]["show_on_image"]:
            self.display_debug_image(anonymized_image, detections)

        # Compute azimuth for each detection
        for i, (bbox, confidence, class_id) in enumerate(
                zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            # Convert to normalized camera coordinates
            x_norm = (x_center - self.cx) / self.fx
            y_norm = (y_center - self.cy) / self.fy

            # Calculate azimuth in degrees
            azimuth_radians = math.atan(x_norm)
            azimuth_degrees = math.degrees(azimuth_radians)

            self.get_logger().info(
                f"Detection {i}: Class={self.DETECTION_CLASSES[class_id]}, "
                f"Confidence={confidence:.2f}, Azimuth={azimuth_degrees:.2f} degrees"
            )


    def camera_info_callback(self, msg):
        self.get_logger().info("Received an image info message")
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(f"Camera intrinsics set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        camera_info_sub = None
        self.destroy_subscription(self.camera_info_sub)

    def anonymize_image(self, image):
        # Detect objects with the Unified Model
        detections = self.unified_language_model(image)

        # Refine detections using SAM2
        detections = self.sam2(image=image, detections=detections)

        # Blur detections
        # output = blur_detections(
        #     image,
        #     detections,
        #     self.config_data["blur"]["kernel_size"],
        #     self.config_data["blur"]["sigma_x"],
        # )
        return image, detections

    def print_detections(self, detections):
        print("\nDetections:")
        for class_id in range(len(self.unified_language_model.detection_classes)):
            print(
                f"{self.unified_language_model.detection_classes[class_id]}: "
                f"{len([d for d in detections if d[3] == class_id])}"
            )

    def display_debug_image(self, image, detections):
        bounding_box_annotator = sv.BoxAnnotator()
        annotated_image = bounding_box_annotator.annotate(
            scene=image,
            detections=detections,
        )

        labels = [
            f"{self.DETECTION_CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]

        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(
            annotated_image,
            detections,
            labels,
        )

        height, width = image.shape[:2]
        annotated_image = cv2.resize(annotated_image, (width // 2, height // 2))
        cv2.imshow("Anonymizer Debug", annotated_image)
        cv2.waitKey(1)


def anonymize_with_unified_model(config_data, json_data, device):
    rclpy.init()
    try:
        node = ImageAnonymizerNode(config_data, json_data, device)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
