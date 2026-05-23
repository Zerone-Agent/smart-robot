import cv2
import numpy as np
import pytest

from src.algorithms.color_segmentation import hsv_red_segmentation, preprocess_image


class TestHSVRedSegmentation:
    def test_hsv_red_segmentation_pure_red(self):
        """Pure red image should have >90% red pixels detected."""
        # Create pure red image in BGR (0, 0, 255 is red in BGR)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [0, 0, 255]

        mask = hsv_red_segmentation(image)
        red_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        ratio = red_pixels / total_pixels

        assert ratio > 0.9, f"Expected >90% red pixels, got {ratio:.2%}"

    def test_hsv_red_segmentation_green_background(self):
        """Pure green should have <100 red pixels."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [0, 255, 0]

        mask = hsv_red_segmentation(image)
        red_pixels = np.count_nonzero(mask)

        assert red_pixels < 100, f"Expected <100 red pixels, got {red_pixels}"


class TestPreprocessImage:
    def test_preprocess_image_shape(self):
        """Output shape should match input when no target_size."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(image)

        assert processed.shape == image.shape

    def test_preprocess_image_resize(self):
        """Should resize when target_size provided."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        target_size = (64, 64)
        processed = preprocess_image(image, target_size=target_size)

        assert processed.shape == (64, 64, 3)
