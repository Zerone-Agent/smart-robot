from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path


MATURITY_MAP = {
    0: 1.0,
    1: 0.6,
    2: 0.2,
}

CLASS_NAMES = {0: "Fully-ripe", 1: "Semi-ripe", 2: "Unripe"}

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (0, 255, 0),
}


class YOLOTomatoDetector:
    def __init__(
        self,
        model_path: str = "external_models/model_hub_n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        from ultralytics import YOLO
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_file}")
        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        result = results[0]
        detections = []
        if result.boxes is None:
            return detections

        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            w = float(x2 - x1)
            h = float(y2 - y1)
            class_id = int(box.cls)
            confidence = float(box.conf)

            cx = float(x1 + w / 2)
            cy = float(y1 + h / 2)
            circularity = min(1.0, (4 * np.pi * (w * h)) / ((2 * (w + h)) ** 2)) if w > 0 and h > 0 else 0.0
            maturity = MATURITY_MAP.get(class_id, 0.5)

            detections.append({
                "tomato_id": i,
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "center_uv": (cx, cy),
                "circularity": circularity,
                "maturity_score": maturity,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, "Unknown"),
                "area": w * h,
            })

        return detections
