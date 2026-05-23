import pytest

from src.algorithms.coordinate_transform import calculate_3d_coordinates


class TestCalculate3DCoordinates:
    def test_calculate_3d_coordinates_center(self):
        """Center point should have X,Y near 0."""
        detections = [
            {
                "center_uv": (320.0, 240.0),
                "bbox": (300, 220, 40, 40),
            }
        ]
        camera_params = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

        results = calculate_3d_coordinates(detections, camera_params, real_diameter_mm=50.0)

        assert len(results) == 1
        pos = results[0]["position_3d"]
        assert abs(pos[0]) < 1.0, f"Expected X near 0, got {pos[0]}"
        assert abs(pos[1]) < 1.0, f"Expected Y near 0, got {pos[1]}"

    def test_calculate_3d_coordinates_offset(self):
        """Right offset should have X > 0."""
        detections = [
            {
                "center_uv": (420.0, 240.0),
                "bbox": (400, 220, 40, 40),
            }
        ]
        camera_params = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

        results = calculate_3d_coordinates(detections, camera_params, real_diameter_mm=50.0)

        assert len(results) == 1
        pos = results[0]["position_3d"]
        assert pos[0] > 0, f"Expected X > 0, got {pos[0]}"
