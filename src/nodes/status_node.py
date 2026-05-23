import time
import threading
from typing import Optional

from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import TomatoDetectionArray, SystemStatus


class StatusNode(MockNode):
    """Monitors detection statistics and publishes system status."""

    def __init__(
        self, publish_interval: float = 5.0, name: str = "status_node"
    ):
        super().__init__(name)
        self.publish_interval = publish_interval

        self.create_subscription(
            "/detection/tomatoes",
            TomatoDetectionArray,
            self._detection_callback,
        )
        self.create_publisher("/system/status", SystemStatus)

        self._lock = threading.Lock()
        self._processed_images = 0
        self._total_detections = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _detection_callback(self, msg: TomatoDetectionArray):
        with self._lock:
            self._processed_images += 1
            if hasattr(msg, 'detections'):
                self._total_detections += len(msg.detections)

    def _run(self):
        while not self._stop_event.is_set():
            with self._lock:
                images = self._processed_images
                detections = self._total_detections
            status = SystemStatus(
                node_name=self.node_name,
                status="RUNNING",
                message=f"Images: {images}, Detections: {detections}",
                timestamp=time.time(),
            )
            self.publish("/system/status", status)
            self.log("INFO", f"Status: {status.message}")

            elapsed = 0.0
            while elapsed < self.publish_interval and not self._stop_event.is_set():
                time.sleep(0.1)
                elapsed += 0.1

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            self.log("WARN", "Node already started")
            return
        super().start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.log("INFO", "Status node started")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        super().stop()
        self.log("INFO", "Status node stopped")
