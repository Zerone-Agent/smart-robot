import os
import time
from typing import Dict, Any, List
import cv2
import numpy as np
import yaml

from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import (
    Image,
    DepthImage,
    TomatoDetection,
    TomatoDetectionArray,
)
from src.algorithms import (
    preprocess_image,
    hsv_tomato_segmentation,
    apply_morphology,
    filter_contours,
    calculate_3d_coordinates,
    load_camera_params,
)


class RecognitionNode(MockNode):
    """Runs tomato detection pipeline on incoming images."""

    def __init__(
        self,
        algorithm_config_path: str = "config/algorithm_params.yaml",
        camera_params_path: str = "data/calibration/camera_params.yaml",
        name: str = "recognition_node",
    ):
        super().__init__(name)
        self.algorithm_config = self._load_yaml(algorithm_config_path)
        self.camera_params = self._load_camera_params(camera_params_path)

        self.create_subscription(
            "/camera/image_raw", Image, self._image_callback
        )
        self.create_subscription(
            "/camera/depth", DepthImage, self._depth_callback
        )
        self.create_publisher("/detection/tomatoes", TomatoDetectionArray)

        self._latest_depth: Dict[str, Any] = {}
        self._image_count = 0

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.log("WARN", f"Config file not found: {path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            self.log("ERROR", f"Invalid YAML in {path}: {e}")
            return {}

    def _load_camera_params(self, path: str) -> Dict[str, float]:
        data = self._load_yaml(path)
        # Handle nested camera_matrix format
        if "camera_matrix" in data:
            cam = data["camera_matrix"]
            return {
                "fx": cam["fx"],
                "fy": cam["fy"],
                "cx": cam["cx"],
                "cy": cam["cy"],
            }
        # Handle flat format
        return load_camera_params(path)

    def _depth_callback(self, msg: DepthImage):
        self._latest_depth = {
            "data": msg.data,
            "width": msg.width,
            "height": msg.height,
            "timestamp": msg.timestamp,
        }

    def _image_callback(self, msg: Image):
        image = msg.data
        timestamp = msg.timestamp

        try:
            detections = self._run_pipeline(image)
            det_array = TomatoDetectionArray(
                detections=detections,
                image_timestamp=timestamp,
            )
            self.publish("/detection/tomatoes", det_array)
            self._image_count += 1
            self.log(
                "INFO",
                f"Detected {len(detections)} tomatoes in image {self._image_count}",
            )
        except Exception as e:
            self.log("ERROR", f"Detection failed: {e}")

    def _run_pipeline(self, image: np.ndarray) -> List[TomatoDetection]:
        # 1. Preprocess
        processed = preprocess_image(image)

        # 2. HSV segmentation (all tomato colors: green, orange, red)
        hsv_params = self.algorithm_config.get("hsv_thresholds", {})
        mask = hsv_tomato_segmentation(
            processed,
            green_lower=tuple(hsv_params.get("green_lower", [35, 50, 40])),
            green_upper=tuple(hsv_params.get("green_upper", [75, 255, 255])),
            red_lower1=tuple(hsv_params.get("lower_red1", [0, 80, 50])),
            red_upper1=tuple(hsv_params.get("upper_red1", [10, 255, 255])),
            red_lower2=tuple(hsv_params.get("lower_red2", [160, 80, 50])),
            red_upper2=tuple(hsv_params.get("upper_red2", [180, 255, 255])),
            orange_lower=tuple(hsv_params.get("orange_lower", [10, 80, 50])),
            orange_upper=tuple(hsv_params.get("orange_upper", [25, 255, 255])),
        )

        # 3. Morphology
        morph_cfg = self.algorithm_config.get("morphology", {})
        mask = apply_morphology(
            mask,
            open_kernel_size=morph_cfg.get("open_kernel_size", 3),
            close_kernel_size=morph_cfg.get("close_kernel_size", 5),
            iterations=morph_cfg.get("iterations", 1),
        )

        # 4. Contour detection and filtering
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_cfg = self.algorithm_config.get("contour_filter", {})
        filtered = filter_contours(
            processed,
            contours,
            min_area=contour_cfg.get("min_area", 100),
            max_area=contour_cfg.get("max_area", 50000),
            min_circularity=contour_cfg.get("min_circularity", 0.5),
            max_aspect_ratio=contour_cfg.get("max_aspect_ratio", 2.0),
        )

        # 5. Calculate 3D coordinates
        depth_cfg = self.algorithm_config.get("depth_estimation", {})
        detections_3d = calculate_3d_coordinates(
            filtered,
            self.camera_params,
            real_diameter_mm=depth_cfg.get("real_diameter_mm", 50.0),
        )

        # Convert to message types
        tomato_detections = []
        for det in detections_3d:
            pos_3d = det.get("position_3d", (0.0, 0.0, 0.0))
            confidence = min(
                (det["circularity"] + det["maturity_score"]) / 2.0, 1.0
            )

            tomato_detections.append(
                TomatoDetection(
                    tomato_id=det["tomato_id"],
                    bbox=list(det["bbox"]),
                    center_uv=list(det["center_uv"]),
                    center_3d=list(pos_3d),
                    circularity=det["circularity"],
                    maturity_score=det["maturity_score"],
                    confidence=confidence,
                )
            )

        return tomato_detections
