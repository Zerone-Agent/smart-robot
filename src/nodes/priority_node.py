from typing import Dict, Any, List

from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import (
    TomatoDetectionArray,
    TomatoTarget,
    TomatoDetection,
)
from src.algorithms import sort_by_priority


class PriorityNode(MockNode):
    """Selects highest priority tomato from detections."""

    def __init__(self, name: str = "priority_node"):
        super().__init__(name)
        self.create_subscription(
            "/detection/tomatoes",
            TomatoDetectionArray,
            self._detection_callback,
        )
        self.create_publisher("/planning/target", TomatoTarget)

    def _detection_callback(self, msg: TomatoDetectionArray):
        if not msg.detections:
            self.log("INFO", "No detections received")
            target = TomatoTarget(detection=None, priority_score=0.0)
            self.publish("/planning/target", target)
            return

        # Convert to dicts for algorithm
        det_dicts = []
        for det in msg.detections:
            det_dict = {
                "tomato_id": det.tomato_id,
                "bbox": det.bbox,
                "center_uv": det.center_uv,
                "center_3d": det.center_3d,
                "circularity": det.circularity,
                "maturity_score": det.maturity_score,
                "confidence": det.confidence,
                "distance": det.center_3d[2] if len(det.center_3d) > 2 else 0.0,
            }
            det_dicts.append(det_dict)

        # Sort by priority
        sorted_dets = sort_by_priority(det_dicts)
        best = sorted_dets[0]

        # Find original detection object
        best_det = None
        for det in msg.detections:
            if det.tomato_id == best["tomato_id"]:
                best_det = det
                break

        target = TomatoTarget(
            detection=best_det,
            priority_score=best.get("priority_score", 0.0),
        )
        self.publish("/planning/target", target)
        self.log(
            "INFO",
            f"Selected target: tomato_id={best['tomato_id']}, score={best.get('priority_score', 0.0):.3f}",
        )
