import os
import time
import threading
from typing import Optional, List
import cv2
import numpy as np

from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import (
    Image,
    TomatoDetectionArray,
    TomatoTarget,
    SystemStatus,
    TomatoDetection,
)


class VisualizationNode(MockNode):
    """Draws detection results and saves visualization images."""

    def __init__(
        self,
        output_dir: str = "results/detection_output",
        name: str = "visualization_node",
    ):
        super().__init__(name)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.create_subscription(
            "/camera/image_raw", Image, self._image_callback
        )
        self.create_subscription(
            "/detection/tomatoes",
            TomatoDetectionArray,
            self._detection_callback,
        )
        self.create_subscription(
            "/planning/target", TomatoTarget, self._target_callback
        )
        self.create_subscription(
            "/system/status", SystemStatus, self._status_callback
        )

        self._lock = threading.Lock()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_detections: List[TomatoDetection] = []
        self._latest_target: Optional[TomatoDetection] = None
        self._latest_status: Optional[SystemStatus] = None
        self._frame_count = 0

    def _image_callback(self, msg: Image):
        with self._lock:
            self._latest_image = msg.data.copy()
        self._draw_and_save()

    def _detection_callback(self, msg: TomatoDetectionArray):
        with self._lock:
            self._latest_detections = msg.detections
        self._draw_and_save()

    def _target_callback(self, msg: TomatoTarget):
        with self._lock:
            self._latest_target = msg.detection

    def _status_callback(self, msg: SystemStatus):
        with self._lock:
            self._latest_status = msg

    def _draw_and_save(self):
        with self._lock:
            image = self._latest_image.copy() if self._latest_image is not None else None
            detections = self._latest_detections
            target = self._latest_target
            status = self._latest_status

        if image is None:
            self.log("DEBUG", "Skip draw: no image received yet")
            return
        if not detections:
            self.log("DEBUG", "Skip draw: no detections")
            return

        h, w = image.shape[:2]

        # Draw detections
        for det in detections:
            x, y, w_box, h_box = det.bbox
            is_target = (
                target is not None
                and det.tomato_id == target.tomato_id
            )

            color = (0, 255, 0) if is_target else (0, 0, 255)
            thickness = 12 if is_target else 8

            # Bounding box
            cv2.rectangle(
                image, (x, y), (x + w_box, y + h_box), color, thickness
            )

            # Center point
            cx, cy = int(det.center_uv[0]), int(det.center_uv[1])
            cv2.circle(image, (cx, cy), 12, color, -1)

            # Label
            label = f"ID:{det.tomato_id} C:{det.confidence:.2f} M:{det.maturity_score:.2f}"
            label_y = max(y - 30, 60)
            cv2.putText(
                image,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                color,
                6,
            )

            # 3D coordinates
            coord_text = f"3D: ({det.center_3d[0]:.0f}, {det.center_3d[1]:.0f}, {det.center_3d[2]:.0f})mm"
            coord_y = min(y + h_box + 60, h - 20)
            cv2.putText(
                image,
                coord_text,
                (x, coord_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                color,
                5,
            )

        # Draw status info
        if status:
            status_text = f"Status: {status.status} | {status.message}"
            cv2.putText(
                image,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                4,
            )

        # Save
        self._frame_count += 1
        filename = f"frame_{self._frame_count:04d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        success = cv2.imwrite(filepath, image)
        if not success:
            self.log("ERROR", f"Failed to save: {filepath}")
        else:
            self.log("INFO", f"Saved visualization: {filename}")
