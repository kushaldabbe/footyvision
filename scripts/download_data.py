"""
Download datasets from Roboflow for FootyVision.

Usage:
    python scripts/download_data.py --api-key YOUR_ROBOFLOW_API_KEY

This downloads:
    1. Football Players Detection dataset (players, goalkeepers, referees, ball)
    2. Soccer Pitch Keypoint dataset (32 pitch landmarks) [Phase 4]
"""

import argparse
from pathlib import Path

from roboflow import Roboflow


def download_player_detection(rf: Roboflow, data_dir: Path) -> None:
    """Download the football-players-detection dataset in YOLOv8 format."""
    print("\n=== Downloading Football Players Detection Dataset ===")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(1)
    dataset = version.download("yolov8", location=str(data_dir / "football-players-detection"))
    print(f"Dataset downloaded to: {dataset.location}")


def main():
    parser = argparse.ArgumentParser(description="Download FootyVision datasets from Roboflow")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument(
        "--data-dir",
        default="data/datasets",
        help="Directory to save datasets (default: data/datasets)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)

    # Phase 0/1: Player detection dataset
    download_player_detection(rf, data_dir)

    # Phase 4: Pitch keypoints (uncomment when ready)
    # download_pitch_keypoints(rf, data_dir)

    print("\n=== All downloads complete ===")


if __name__ == "__main__":
    main()
