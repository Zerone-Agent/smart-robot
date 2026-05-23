#!/usr/bin/env python3
"""Generate analysis report from YOLO detection results."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms import (
    YOLOTomatoDetector,
    calculate_3d_coordinates,
    load_camera_params,
)
from src.algorithms.yolo_detection import CLASS_NAMES, MATURITY_MAP


def analyze_image(detector: YOLOTomatoDetector, image_path: str, camera_params: Dict) -> Dict[str, Any]:
    image = cv2.imread(image_path)
    if image is None:
        return {"filename": os.path.basename(image_path), "detections": 0, "error": "Failed to load"}

    detections = detector.detect(image)

    depth_cfg = {"real_diameter_mm": 80.0}
    detections_3d = calculate_3d_coordinates(
        detections,
        camera_params,
        real_diameter_mm=depth_cfg.get("real_diameter_mm", 80.0),
    )

    confidences = []
    maturities = []
    distances = []
    class_counts = {"Fully-ripe": 0, "Semi-ripe": 0, "Unripe": 0}

    for det in detections_3d:
        confidences.append(det["confidence"])
        maturities.append(det["maturity_score"])
        pos = det.get("position_3d", (0, 0, 0))
        distances.append(pos[2] if len(pos) > 2 else 0)
        name = CLASS_NAMES.get(det.get("class_id", 0), "Unknown")
        if name in class_counts:
            class_counts[name] += 1

    return {
        "filename": os.path.basename(image_path),
        "detections": len(detections_3d),
        "confidences": confidences,
        "maturities": maturities,
        "distances": distances,
        "class_counts": class_counts,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_maturity": float(np.mean(maturities)) if maturities else 0.0,
        "avg_distance": float(np.mean(distances)) if distances else 0.0,
    }


def create_sample_grid(image_paths: List[str], output_path: str, grid_size: int = 3):
    if len(image_paths) < grid_size * grid_size:
        print(f"[WARN] Only {len(image_paths)} images available")
        return

    step = len(image_paths) // (grid_size * grid_size)
    selected = [image_paths[i * step] for i in range(grid_size * grid_size)]

    thumb_w = 720
    thumb_h = 960
    padding = 4
    border_color = (80, 80, 80)

    images = []
    for path in selected:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (thumb_w, thumb_h))
            img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=border_color)
            images.append(img)

    if not images:
        return

    rows = []
    for i in range(grid_size):
        row = np.hstack(images[i * grid_size : (i + 1) * grid_size])
        rows.append(row)
    grid = np.vstack(rows)

    cv2.imwrite(output_path, grid)
    print(f"[INFO] Saved sample grid: {output_path}")


def create_summary_image(report: Dict, output_path: str):
    w, h = 1200, 900
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.putText(img, "Tomato Detection Report", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    cv2.line(img, (40, 90), (w - 40, 90), (0, 0, 0), 2)

    summary = report["summary"]
    y = 150
    dy = 55
    fs = 1.0
    th = 2

    items = [
        ("Total Images Processed", str(summary["total_images_processed"])),
        ("Total Tomatoes Detected", str(summary["total_tomatoes_detected"])),
        ("Avg Detections / Image", str(summary["avg_detections_per_image"])),
        ("Avg Confidence", f'{summary["overall_avg_confidence"]:.1%}'),
        ("Avg Maturity Score", f'{summary["overall_avg_maturity"]:.3f}'),
        ("Avg Distance (mm)", f'{summary["overall_avg_distance_mm"]:.1f}'),
    ]

    for label, value in items:
        cv2.putText(img, f"{label}:", (60, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), th)
        cv2.putText(img, value, (550, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th)
        y += dy

    y += 20
    cv2.putText(img, "Class Distribution", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    y += 50

    dist = summary["class_distribution"]
    total_cls = sum(dist.values())

    bar_x = 60
    bar_w = 600
    bar_h = 40

    colors = {
        "Fully-ripe": (0, 0, 220),
        "Semi-ripe": (0, 165, 255),
        "Unripe": (0, 200, 0),
    }
    labels_cn = {"Fully-ripe": "Fully-ripe", "Semi-ripe": "Semi-ripe", "Unripe": "Unripe"}

    cx = bar_x
    for cls in ["Fully-ripe", "Semi-ripe", "Unripe"]:
        cnt = dist.get(cls, 0)
        bw = int(bar_w * cnt / total_cls) if total_cls > 0 else 0
        cv2.rectangle(img, (cx, y), (cx + bw, y + bar_h), colors[cls], -1)
        cx += bw

    y += bar_h + 15
    for cls in ["Fully-ripe", "Semi-ripe", "Unripe"]:
        cnt = dist.get(cls, 0)
        pct = cnt / total_cls * 100 if total_cls > 0 else 0
        cv2.rectangle(img, (60, y), (90, y + 20), colors[cls], -1)
        cv2.putText(img, f"{labels_cn[cls]}: {cnt} ({pct:.1f}%)", (100, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 35

    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved summary image: {output_path}")


def main():
    base_dir = Path(__file__).parent.parent.parent
    detection_dir = base_dir / "results" / "detection_output"
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    config_path = base_dir / "config" / "algorithm_params.yaml"
    camera_path = base_dir / "data" / "calibration" / "camera_params.yaml"
    data_dir = base_dir / "data" / "DatasetId_360176_1652587113" / "Images"

    with open(config_path, "r") as f:
        algorithm_config = yaml.safe_load(f) or {}

    with open(camera_path, "r") as f:
        cam_data = yaml.safe_load(f) or {}
    if "camera_matrix" in cam_data:
        cam = cam_data["camera_matrix"]
        camera_params = {"fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"]}
    else:
        camera_params = load_camera_params(str(camera_path))

    yolo_cfg = algorithm_config.get("yolo", {})
    detector = YOLOTomatoDetector(
        model_path=yolo_cfg.get("model_path", "external_models/model_hub_n.pt"),
        conf_threshold=yolo_cfg.get("conf_threshold", 0.5),
        iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
    )

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(data_dir.rglob(ext))
    image_paths = sorted([str(p) for p in image_paths])

    if not image_paths:
        print("[ERROR] No test images found")
        sys.exit(1)

    print(f"[INFO] Analyzing {len(image_paths)} images...")

    results = []
    for i, path in enumerate(image_paths, 1):
        result = analyze_image(detector, path, camera_params)
        results.append(result)
        if i % 10 == 0:
            print(f"  [{i}/{len(image_paths)}] processed")

    total_images = len(results)
    total_detections = sum(r["detections"] for r in results)
    avg_detections = total_detections / total_images if total_images > 0 else 0

    all_confidences = [c for r in results for c in r["confidences"]]
    all_maturities = [m for r in results for m in r["maturities"]]
    all_distances = [d for r in results for d in r["distances"]]

    total_class_counts = {"Fully-ripe": 0, "Semi-ripe": 0, "Unripe": 0}
    for r in results:
        for cls, cnt in r.get("class_counts", {}).items():
            total_class_counts[cls] += cnt

    report = {
        "summary": {
            "total_images_processed": total_images,
            "total_tomatoes_detected": total_detections,
            "avg_detections_per_image": round(avg_detections, 2),
            "overall_avg_confidence": round(float(np.mean(all_confidences)), 3) if all_confidences else 0.0,
            "overall_avg_maturity": round(float(np.mean(all_maturities)), 3) if all_maturities else 0.0,
            "overall_avg_distance_mm": round(float(np.mean(all_distances)), 1) if all_distances else 0.0,
            "class_distribution": total_class_counts,
        },
        "per_image_details": sorted(
            [
                {
                    "filename": r["filename"],
                    "detections": r["detections"],
                    "avg_confidence": round(r["avg_confidence"], 3),
                    "avg_maturity": round(r["avg_maturity"], 3),
                    "class_counts": r.get("class_counts", {}),
                }
                for r in results
            ],
            key=lambda x: x["avg_confidence"],
            reverse=True,
        ),
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved report: {report_path}")

    viz_paths = sorted(detection_dir.glob("frame_*.jpg"))
    if viz_paths:
        grid_path = output_dir / "detection_grid.jpg"
        create_sample_grid([str(p) for p in viz_paths], str(grid_path))

    summary_path = output_dir / "summary_report.jpg"
    create_summary_image(report, str(summary_path))

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()
