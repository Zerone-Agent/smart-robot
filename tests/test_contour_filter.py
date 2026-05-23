import cv2
import numpy as np
import pytest

from src.algorithms.contour_filter import calculate_circularity, filter_contours


class TestCalculateCircularity:
    def test_circularity_perfect_circle(self):
        """Circle should have circularity >0.85."""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(image, (100, 100), 50, 255, -1)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0

        circularity = calculate_circularity(contours[0])
        assert circularity > 0.85, f"Expected circularity >0.85, got {circularity}"

    def test_circularity_rectangle(self):
        """Rectangle should have circularity <0.8."""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 100), 255, -1)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0

        circularity = calculate_circularity(contours[0])
        assert circularity < 0.8, f"Expected circularity <0.8, got {circularity}"


class TestFilterContours:
    def test_filter_contours_min_area(self):
        """Should filter out small contours."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create a large red circle
        cv2.circle(image, (100, 100), 50, (0, 0, 255), -1)
        # Create a small red dot
        cv2.circle(image, (30, 30), 3, (0, 0, 255), -1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = filter_contours(image, contours, min_area=100)

        # Should only keep the large circle
        assert len(filtered) >= 1
        for det in filtered:
            assert det["area"] >= 100
