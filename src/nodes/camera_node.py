import os
import time
import threading
from typing import Optional, List
import cv2
import numpy as np

from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import Image, DepthImage


class CameraNode(MockNode):
    """Publishes images and synthetic depth maps from a directory."""

    def __init__(
        self,
        image_dir: str = "data/test_images",
        publish_rate: float = 1.0,
        name: str = "camera_node",
    ):
        super().__init__(name)
        self.image_dir = image_dir
        self.publish_rate = publish_rate
        self.image_paths: List[str] = []
        self.current_index = 0
        self._load_images()

        self.create_publisher("/camera/image_raw", Image)
        self.create_publisher("/camera/depth", DepthImage)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _load_images(self):
        """Recursively load all image paths from directory."""
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        if os.path.exists(self.image_dir):
            for root, _, files in os.walk(self.image_dir):
                for f in sorted(files):
                    if f.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(root, f))
        self.log("INFO", f"Loaded {len(self.image_paths)} images from {self.image_dir}")

    def _generate_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate synthetic depth map."""
        h, w = image.shape[:2]
        depth = np.full((h, w), 1200.0, dtype=np.float32)
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        depth_variation = 200 * (1 - dist / max_dist)
        depth = depth - depth_variation
        noise = np.random.normal(0, 20, (h, w))
        depth = np.clip(depth + noise, 500, 2000)
        return depth.astype(np.float32)

    def _publish_frame(self):
        """Load and publish current frame."""
        if not self.image_paths:
            self.log("WARN", "No images to publish")
            return

        path = self.image_paths[self.current_index]
        image = cv2.imread(path)
        if image is None:
            self.log("ERROR", f"Failed to load image: {path}")
            return

        height, width, channels = image.shape
        timestamp = time.time()

        img_msg = Image(
            data=image,
            width=width,
            height=height,
            channels=channels,
            timestamp=timestamp,
        )
        self.publish("/camera/image_raw", img_msg)

        depth_data = self._generate_depth(image)
        depth_msg = DepthImage(
            data=depth_data,
            width=width,
            height=height,
            timestamp=timestamp,
        )
        self.publish("/camera/depth", depth_msg)

        self.log(
            "INFO",
            f"Published frame {self.current_index + 1}/{len(self.image_paths)}: {os.path.basename(path)}",
        )
        self.current_index = (self.current_index + 1) % len(self.image_paths)

    def _run(self):
        """Main publishing loop."""
        interval = 1.0 / self.publish_rate
        while not self._stop_event.is_set():
            self._publish_frame()
            time.sleep(interval)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            self.log("WARN", "Node already started")
            return
        super().start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.log("INFO", "Camera node started")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        super().stop()
        self.log("INFO", "Camera node stopped")
