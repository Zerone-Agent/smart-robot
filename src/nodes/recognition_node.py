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
    YOLOTomatoDetector,
    calculate_3d_coordinates,
    load_camera_params,
)


class RecognitionNode(MockNode):
    """Runs tomato detection pipeline on incoming images using YOLO model."""

    def __init__(
        self,
        algorithm_config_path: str = "config/algorithm_params.yaml",
        camera_params_path: str = "data/calibration/camera_params.yaml",
        name: str = "recognition_node",
    ):
        super().__init__(name)
        self.algorithm_config = self._load_yaml(algorithm_config_path)
        self.camera_params = self._load_camera_params(camera_params_path)

        yolo_cfg = self.algorithm_config.get("yolo", {})
        self.detector = YOLOTomatoDetector(
            model_path=yolo_cfg.get("model_path", "external_models/model_hub_n.pt"),
            conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
            iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
        )

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
        if "camera_matrix" in data:
            cam = data["camera_matrix"]
            return {
                "fx": cam["fx"],
                "fy": cam["fy"],
                "cx": cam["cx"],
                "cy": cam["cy"],
            }
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
        yolo_detections = self.detector.detect(image)

        depth_cfg = self.algorithm_config.get("depth_estimation", {})
        detections_3d = calculate_3d_coordinates(
            yolo_detections,
            self.camera_params,
            real_diameter_mm=depth_cfg.get("real_diameter_mm", 50.0),
        )

        tomato_detections = []
        for det in detections_3d:
            pos_3d = det.get("position_3d", (0.0, 0.0, 0.0))
            tomato_detections.append(
                TomatoDetection(
                    tomato_id=det["tomato_id"],
                    bbox=list(det["bbox"]),
                    center_uv=list(det["center_uv"]),
                    center_3d=list(pos_3d),
                    circularity=det["circularity"],
                    maturity_score=det["maturity_score"],
                    confidence=det["confidence"],
                )
            )

        return tomato_detections
