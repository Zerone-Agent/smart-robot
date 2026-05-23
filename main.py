#!/usr/bin/env python3
"""Main entry point for the smart robot tomato picking system."""

import argparse
import os
import sys
import time
from typing import Optional

import yaml

from src.mock_ros2.core import MockSystem, MockNode
from src.mock_ros2.message_types import TomatoDetectionArray
from src.nodes import (
    CameraNode,
    RecognitionNode,
    PriorityNode,
    VisualizationNode,
    StatusNode,
)


class ImageCounter(MockNode):
    """Simple node to count processed images for max_images limit."""

    def __init__(self, max_images: Optional[int] = None):
        super().__init__("image_counter")
        self.max_images = max_images
        self.count = 0
        self.create_subscription(
            "/detection/tomatoes", TomatoDetectionArray, self._callback
        )

    def _callback(self, msg: TomatoDetectionArray):
        self.count += 1


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML in {path}: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Robot Tomato Picking System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/DatasetId_360176_1652587113/Images",
        help="Test image directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/algorithm_params.yaml",
        help="Algorithm parameters YAML file",
    )
    parser.add_argument(
        "--camera-params",
        type=str,
        default="data/calibration/camera_params.yaml",
        help="Camera parameters YAML file",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Publish rate in Hz",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to process (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate paths
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Load configurations
    algorithm_params = load_yaml(args.config)
    camera_params = load_yaml(args.camera_params)

    print(f"[INFO] Loaded algorithm config from: {args.config}")
    print(f"[INFO] Loaded camera params from: {args.camera_params}")
    print(f"[INFO] Image directory: {args.data_dir}")
    print(f"[INFO] Publish rate: {args.rate} Hz")
    if args.max_images:
        print(f"[INFO] Max images to process: {args.max_images}")

    # Create system
    system = MockSystem()

    # Create nodes
    camera_node = CameraNode(
        image_dir=args.data_dir,
        publish_rate=args.rate,
    )
    recognition_node = RecognitionNode(
        algorithm_config_path=args.config,
        camera_params_path=args.camera_params,
    )
    priority_node = PriorityNode()
    visualization_node = VisualizationNode()
    status_node = StatusNode()

    # Register nodes
    system.register_node(camera_node)
    system.register_node(recognition_node)
    system.register_node(priority_node)
    system.register_node(visualization_node)
    system.register_node(status_node)

    # Add image counter if max_images is set
    image_counter = None
    if args.max_images:
        image_counter = ImageCounter(max_images=args.max_images)
        system.register_node(image_counter)

    # Start all nodes
    print("[INFO] Starting system...")
    system.start_all()

    try:
        # Main loop
        while True:
            time.sleep(0.1)
            if image_counter and image_counter.count >= args.max_images:
                print(f"[INFO] Reached max images limit: {args.max_images}")
                break
            if not camera_node._thread.is_alive():
                print("[INFO] Camera node finished, all images processed")
                break
    except KeyboardInterrupt:
        print("\n[INFO] Received KeyboardInterrupt, shutting down...")

    # Stop all nodes gracefully
    print("[INFO] Stopping all nodes...")
    system.stop_all()

    # Give nodes time to finish
    time.sleep(0.5)

    # Print summary
    print("\n" + "=" * 50)
    print("SYSTEM SHUTDOWN COMPLETE")
    print("=" * 50)
    print(f"Images processed: {image_counter.count if image_counter else 'N/A (unlimited)'}")
    print(f"Visualization results saved to: {visualization_node.output_dir}")
    print(f"Detection results: {recognition_node._image_count} images processed")
    print("=" * 50)


if __name__ == "__main__":
    main()
