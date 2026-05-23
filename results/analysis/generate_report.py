#!/usr/bin/env python3
"""Generate analysis report from detection results."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms import (
    preprocess_image,
    hsv_red_segmentation,
    apply_morphology,
    filter_contours,
    calculate_3d_coordinates,
    load_camera_params,
)


def analyze_image(image_path: str, algorithm_config: Dict, camera_params: Dict) -> Dict[str, Any]:
    """Run detection pipeline on a single image and return statistics."""
    image = cv2.imread(image_path)
    if image is None:
        return {"filename": os.path.basename(image_path), "detections": 0, "error": "Failed to load"}

    # Run pipeline
    processed = preprocess_image(image)
    hsv_params = algorithm_config.get("hsv_thresholds", {})
    mask = hsv_red_segmentation(
        processed,
        lower1=tuple(hsv_params.get("lower_red1", [0, 80, 50])),
        upper1=tuple(hsv_params.get("upper_red1", [10, 255, 255])),
        lower2=tuple(hsv_params.get("lower_red2", [160, 80, 50])),
        upper2=tuple(hsv_params.get("upper_red2", [180, 255, 255])),
    )

    morph_cfg = algorithm_config.get("morphology", {})
    mask = apply_morphology(
        mask,
        open_kernel_size=morph_cfg.get("open_kernel_size", 5),
        close_kernel_size=morph_cfg.get("close_kernel_size", 7),
        iterations=morph_cfg.get("iterations", 2),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_cfg = algorithm_config.get("contour_filter", {})
    filtered = filter_contours(
        processed,
        contours,
        min_area=contour_cfg.get("min_area", 500),
        max_area=contour_cfg.get("max_area", 50000),
        min_circularity=contour_cfg.get("min_circularity", 0.5),
        max_aspect_ratio=contour_cfg.get("max_aspect_ratio", 2.0),
    )

    depth_cfg = algorithm_config.get("depth_estimation", {})
    detections_3d = calculate_3d_coordinates(
        filtered,
        camera_params,
        real_diameter_mm=depth_cfg.get("real_diameter_mm", 80.0),
    )

    # Extract stats
    confidences = []
    maturities = []
    distances = []

    for det in detections_3d:
        conf = min((det["circularity"] + det["maturity_score"]) / 2.0, 1.0)
        confidences.append(conf)
        maturities.append(det["maturity_score"])
        pos = det.get("position_3d", (0, 0, 0))
        distances.append(pos[2] if len(pos) > 2 else 0)

    # Determine scene type from path
    scene = "unknown"
    for s in ["single_tomato", "multiple_tomatoes", "occlusion", "lighting"]:
        if s in image_path:
            scene = s
            break

    return {
        "filename": os.path.basename(image_path),
        "scene": scene,
        "detections": len(detections_3d),
        "confidences": confidences,
        "maturities": maturities,
        "distances": distances,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_maturity": float(np.mean(maturities)) if maturities else 0.0,
        "avg_distance": float(np.mean(distances)) if distances else 0.0,
    }


def create_sample_grid(image_paths: List[str], output_path: str, grid_size: int = 3):
    """Create a grid image showing example detections."""
    if len(image_paths) < 9:
        print(f"Warning: Only {len(image_paths)} images available, skipping grid")
        return
    selected = image_paths[: grid_size * grid_size]
    if len(selected) < grid_size * grid_size:
        # Repeat if not enough
        selected = selected * (grid_size * grid_size // len(selected) + 1)
        selected = selected[: grid_size * grid_size]

    images = []
    for path in selected:
        img = cv2.imread(path)
        if img is not None:
            # Resize to uniform size
            img = cv2.resize(img, (400, 300))
            images.append(img)

    if not images:
        print("[WARN] No images to create grid")
        return

    # Create grid
    rows = []
    for i in range(grid_size):
        row = np.hstack(images[i * grid_size : (i + 1) * grid_size])
        rows.append(row)
    grid = np.vstack(rows)

    # Add title
    title = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title,
        "Detection Results Sample Grid",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    grid = np.vstack([title, grid])

    cv2.imwrite(output_path, grid)
    print(f"[INFO] Saved sample grid: {output_path}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    detection_dir = base_dir / "results" / "detection_output"
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = base_dir / "config" / "algorithm_params.yaml"
    camera_path = base_dir / "data" / "calibration" / "camera_params.yaml"
    data_dir = base_dir / "data" / "DatasetId_360176_1652587113" / "Images"

    # Load configs
    import yaml

    with open(config_path, "r") as f:
        algorithm_config = yaml.safe_load(f) or {}
    # Load camera params (handle nested format)
    with open(camera_path, "r") as f:
        cam_data = yaml.safe_load(f) or {}
    if "camera_matrix" in cam_data:
        cam = cam_data["camera_matrix"]
        camera_params = {"fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"]}
    else:
        camera_params = load_camera_params(str(camera_path))

    # Gather all test images
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(data_dir.rglob(ext))
    image_paths = sorted([str(p) for p in image_paths])

    if not image_paths:
        print("[ERROR] No test images found")
        sys.exit(1)

    print(f"[INFO] Analyzing {len(image_paths)} images...")

    # Analyze each image
    results = []
    for path in image_paths:
        result = analyze_image(path, algorithm_config, camera_params)
        results.append(result)

    # Aggregate statistics
    total_images = len(results)
    total_detections = sum(r["detections"] for r in results)
    avg_detections = total_detections / total_images if total_images > 0 else 0

    all_confidences = [c for r in results for c in r["confidences"]]
    all_maturities = [m for r in results for m in r["maturities"]]
    all_distances = [d for r in results for d in r["distances"]]

    # Scene breakdown
    scene_stats = {}
    for r in results:
        scene = r["scene"]
        if scene not in scene_stats:
            scene_stats[scene] = {"images": 0, "detections": 0, "avg_confidence": []}
        scene_stats[scene]["images"] += 1
        scene_stats[scene]["detections"] += r["detections"]
        scene_stats[scene]["avg_confidence"].extend(r["confidences"])

    for scene in scene_stats:
        confs = scene_stats[scene]["avg_confidence"]
        scene_stats[scene]["avg_confidence"] = round(float(np.mean(confs)), 3) if confs else 0.0
        scene_stats[scene]["avg_detections_per_image"] = (
            scene_stats[scene]["detections"] / scene_stats[scene]["images"]
        )

    report = {
        "summary": {
            "total_images_processed": total_images,
            "total_tomatoes_detected": total_detections,
            "avg_detections_per_image": round(avg_detections, 2),
            "overall_avg_confidence": round(float(np.mean(all_confidences)), 3) if all_confidences else 0.0,
            "overall_avg_maturity": round(float(np.mean(all_maturities)), 3) if all_maturities else 0.0,
            "overall_avg_distance_mm": round(float(np.mean(all_distances)), 1) if all_distances else 0.0,
        },
        "scene_breakdown": scene_stats,
        "top_detections": sorted(
            [
                {
                    "filename": r["filename"],
                    "scene": r["scene"],
                    "detections": r["detections"],
                    "avg_confidence": round(r["avg_confidence"], 3),
                    "avg_maturity": round(r["avg_maturity"], 3),
                }
                for r in results
            ],
            key=lambda x: x["avg_confidence"],
            reverse=True,
        )[:10],
    }

    # Save JSON report
    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved report: {report_path}")

    # Create sample grid from detection outputs
    viz_paths = sorted(detection_dir.glob("*.jpg"))
    if not viz_paths:
        print("[WARN] No visualization images found, using test images")
        viz_paths = [Path(p) for p in image_paths]

    grid_path = output_dir / "detection_grid.jpg"
    create_sample_grid([str(p) for p in viz_paths], str(grid_path))

    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()
