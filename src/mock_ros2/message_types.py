from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Image:
    data: np.ndarray
    width: int
    height: int
    channels: int
    timestamp: float = 0.0


@dataclass
class DepthImage:
    data: np.ndarray
    width: int
    height: int
    timestamp: float = 0.0


@dataclass
class TomatoDetection:
    tomato_id: int
    bbox: List[int]          # [x, y, w, h]
    center_uv: List[float]   # [u, v]
    center_3d: List[float]   # [X, Y, Z] mm
    circularity: float
    maturity_score: float
    confidence: float
    class_id: int = 0        # 0=Fully-ripe, 1=Semi-ripe, 2=Unripe


@dataclass
class TomatoDetectionArray:
    detections: List[TomatoDetection] = field(default_factory=list)
    image_timestamp: float = 0.0


@dataclass
class TomatoTarget:
    detection: Optional[TomatoDetection] = None
    priority_score: float = 0.0


@dataclass
class SystemStatus:
    node_name: str
    status: str
    message: str
    timestamp: float = 0.0
