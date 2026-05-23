import cv2
import numpy as np


def apply_morphology(
    mask: np.ndarray,
    open_kernel_size: int = 5,
    close_kernel_size: int = 7,
    iterations: int = 2,
) -> np.ndarray:
    """Apply morphological open then close to binary mask."""
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    return closed
