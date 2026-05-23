import cv2
import numpy as np
import pytest

from src.algorithms import (
    preprocess_image,
    hsv_red_segmentation,
    apply_morphology,
    filter_contours,
    calculate_3d_coordinates,
)


class TestFullPipeline:
    def test_full_pipeline_single_tomato(self):
        """Create image with red circle, run full pipeline, verify detection."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Green background
        image[:, :] = [0, 128, 0]
        # Red circle (tomato)
        cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)

        # 1. Preprocess
        processed = preprocess_image(image)

        # 2. HSV segmentation
        mask = hsv_red_segmentation(processed)

        # 3. Morphology
        mask = apply_morphology(mask)

        # 4. Contour detection and filtering
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = filter_contours(processed, contours, min_area=100)

        # 5. 3D coordinates
        camera_params = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        detections_3d = calculate_3d_coordinates(detections, camera_params, real_diameter_mm=50.0)

        assert len(detections_3d) >= 1, "Expected at least 1 detection"

        # Verify the detected tomato is near the center
        det = detections_3d[0]
        center_u, center_v = det["center_uv"]
        assert abs(center_u - 320) < 10, f"Expected center_u near 320, got {center_u}"
        assert abs(center_v - 240) < 10, f"Expected center_v near 240, got {center_v}"

    def test_full_pipeline_no_tomato(self):
        """Pure green image should have 0 detections."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :] = [0, 255, 0]

        processed = preprocess_image(image)
        mask = hsv_red_segmentation(processed)
        mask = apply_morphology(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = filter_contours(processed, contours, min_area=100)

        camera_params = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        detections_3d = calculate_3d_coordinates(detections, camera_params, real_diameter_mm=50.0)

        assert len(detections_3d) == 0, f"Expected 0 detections, got {len(detections_3d)}"
