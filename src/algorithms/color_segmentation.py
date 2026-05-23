from typing import Optional, Tuple
import cv2
import numpy as np


def hsv_red_segmentation(
    image: np.ndarray,
    lower1: Tuple[int, int, int] = (0, 80, 50),
    upper1: Tuple[int, int, int] = (10, 255, 255),
    lower2: Tuple[int, int, int] = (160, 80, 50),
    upper2: Tuple[int, int, int] = (180, 255, 255),
) -> np.ndarray:
    """Convert BGR image to HSV and create mask for red color ranges."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
    mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
    mask = cv2.bitwise_or(mask1, mask2)
    return mask


def hsv_tomato_segmentation(
    image: np.ndarray,
    green_lower: Tuple[int, int, int] = (25, 40, 30),
    green_upper: Tuple[int, int, int] = (85, 255, 255),
    red_lower1: Tuple[int, int, int] = (0, 80, 50),
    red_upper1: Tuple[int, int, int] = (10, 255, 255),
    red_lower2: Tuple[int, int, int] = (160, 80, 50),
    red_upper2: Tuple[int, int, int] = (180, 255, 255),
    orange_lower: Tuple[int, int, int] = (10, 80, 50),
    orange_upper: Tuple[int, int, int] = (25, 255, 255),
) -> np.ndarray:
    """
    Segment all tomato colors: green (unripe), orange/yellow (ripening), and red (ripe).
    
    Returns a combined mask covering all tomato maturity stages.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Green tomatoes (unripe)
    green_mask = cv2.inRange(hsv, np.array(green_lower), np.array(green_upper))
    
    # Red tomatoes (ripe) - two ranges for hue wrap-around
    red_mask1 = cv2.inRange(hsv, np.array(red_lower1), np.array(red_upper1))
    red_mask2 = cv2.inRange(hsv, np.array(red_lower2), np.array(red_upper2))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Orange/yellow tomatoes (ripening)
    orange_mask = cv2.inRange(hsv, np.array(orange_lower), np.array(orange_upper))
    
    # Combine all tomato colors
    combined = cv2.bitwise_or(green_mask, red_mask)
    combined = cv2.bitwise_or(combined, orange_mask)
    
    return combined


def preprocess_image(
    image: np.ndarray, target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Apply Gaussian blur, CLAHE, and optional resize."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l)
    lab_equalized = cv2.merge([l_equalized, a, b])
    result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    if target_size is not None:
        result = cv2.resize(result, target_size)
    return result
