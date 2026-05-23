import os
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple


def calculate_3d_coordinates(
    detections: List[Dict[str, Any]],
    camera_params: Dict[str, float],
    real_diameter_mm: float,
) -> List[Dict[str, Any]]:
    """Calculate 3D coordinates from 2D detections."""
    if real_diameter_mm <= 0:
        raise ValueError(f"real_diameter_mm must be > 0, got {real_diameter_mm}")
    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    results = []
    for det in detections:
        u, v = det["center_uv"]
        bbox = det["bbox"]
        pixel_diameter = max(bbox[2], bbox[3])
        if pixel_diameter == 0:
            continue
        Z = fx * real_diameter_mm / pixel_diameter
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        det_3d = det.copy()
        det_3d["position_3d"] = (X, Y, Z)
        results.append(det_3d)
    return results


def load_camera_params(filepath: str) -> Dict[str, float]:
    """Load camera parameters from YAML file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Camera params file not found: {filepath}")
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    required_keys = ["fx", "fy", "cx", "cy"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required camera parameter: {key}")
    return {
        "fx": data["fx"],
        "fy": data["fy"],
        "cx": data["cx"],
        "cy": data["cy"],
    }
