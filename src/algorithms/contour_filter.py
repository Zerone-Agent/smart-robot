import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def calculate_circularity(contour: np.ndarray) -> float:
    """Calculate circularity: C = 4*pi*A/P^2."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def calculate_maturity_score(
    hsv: np.ndarray, red_mask: np.ndarray, contour: np.ndarray
) -> float:
    """Calculate maturity score based on red pixel ratio and saturation."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    tomato_mask = cv2.bitwise_and(mask, red_mask)
    total_pixels = cv2.countNonZero(mask)
    red_pixels = cv2.countNonZero(tomato_mask)
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
    saturation = hsv[:, :, 1]
    avg_saturation = cv2.mean(saturation, mask=mask)[0]
    maturity = red_ratio * 0.7 + (avg_saturation / 255.0) * 0.3
    return maturity


def filter_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    min_area: int = 100,
    max_area: int = 50000,
    min_circularity: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> List[Dict[str, Any]]:
    """Filter contours by area, circularity, and aspect ratio."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 80, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    filtered = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        circularity = calculate_circularity(contour)
        if circularity < min_circularity:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float("inf")
        if aspect_ratio > max_aspect_ratio:
            continue
        center_u = x + w / 2
        center_v = y + h / 2
        maturity = calculate_maturity_score(hsv, red_mask, contour)
        detection = {
            "tomato_id": idx,
            "bbox": (x, y, w, h),
            "center_uv": (center_u, center_v),
            "circularity": circularity,
            "maturity_score": maturity,
            "area": area,
            "contour": contour,
        }
        filtered.append(detection)
    return filtered
