"""
06_train_pitch_keypoints.py — Fine-tune YOLOv8-pose for Pitch Keypoint Detection

=== WHAT THIS SCRIPT DOES ===
Fine-tunes a YOLOv8n-pose (nano) model to detect 32 pitch landmarks from broadcast
football footage. These keypoints are used to compute a homography transform
for mapping player positions to a top-down tactical view.

=== WHY YOLOv8-POSE FOR KEYPOINTS? ===

YOLOv8-pose was designed for human pose estimation (17 body keypoints), but the
architecture is general-purpose: detect bounding boxes with associated keypoints.
We repurpose it for pitch landmark detection:
  - Input: broadcast football image
  - Output: one bounding box (the pitch) + 32 keypoints (landmarks)

Each keypoint has 3 values: (x, y, visibility)
  - x, y: pixel coordinates in the image
  - visibility: 0 = not labeled, 1 = labeled but occluded, 2 = labeled and visible

Not all 32 keypoints are visible in every frame (broadcast cameras show ~30-50%
of the pitch at once). The model learns to predict visibility alongside position,
so downstream code can filter to only use confident visible keypoints.

=== KEYPOINT DETECTION HEAD ===

Standard YOLOv8 detection head outputs: [x, y, w, h, conf, class_scores]
YOLOv8-pose adds: [kpt1_x, kpt1_y, kpt1_conf, kpt2_x, kpt2_y, kpt2_conf, ...]

For 32 keypoints: 32 × 3 = 96 extra outputs per detection.
The keypoint loss is OKS (Object Keypoint Similarity) — same metric used in COCO.

=== WHY YOLOv8n-POSE (NANO) INSTEAD OF YOLOv8s-POSE? ===

Our GTX 1650Ti has 4GB VRAM. The pose head adds significant memory:
  - yolov8s-pose with 32 keypoints: ~2.5GB (tight with batch > 2)
  - yolov8n-pose with 32 keypoints: ~1.5GB (batch=4-8 works)

Also, pitch keypoint detection is easier than human pose — the pitch is
a rigid, flat surface with high-contrast markings. Nano is sufficient.

=== AV/ROBOTICS PARALLEL ===
This is MAP FEATURE DETECTION — detecting known landmarks in the environment.
In AV:
  - Detect lane markings, traffic signs, curbs from camera images
  - Match detected features to HD Map landmarks
  - Compute camera pose (localization) from the correspondences
Our pipeline does the same: detect pitch markings → match to pitch template →
compute camera-to-pitch transform (homography).

Usage:
    python scripts/06_train_pitch_keypoints.py
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    # === PATHS ===
    data_yaml = Path("data/datasets/football-pitch-keypoints/data.yaml")

    if not data_yaml.exists():
        print(f"ERROR: Dataset not found at {data_yaml}")
        print("Download it first:")
        print("  python scripts/download_data.py --api-key YOUR_KEY")
        return

    # === MODEL ===
    # YOLOv8n-pose: nano variant with pose/keypoint head
    # "yolov8n-pose.pt" = pretrained on COCO keypoints (17 body keypoints)
    # Fine-tuning will adapt it to our 32 pitch keypoints
    model = YOLO("yolov8n-pose.pt")

    # === TRAINING ===
    results = model.train(
        data=str(data_yaml),

        # --- Training duration ---
        epochs=100,         # Enough for convergence on 317 images
        patience=20,        # Early stop if no improvement for 20 epochs

        # --- Input & batch ---
        imgsz=640,          # Standard YOLO input resolution
        batch=4,            # Conservative for 4GB VRAM with 32-keypoint head
        # If OOM: try batch=2 or batch=1

        # --- Optimization ---
        lr0=0.001,          # Lower LR than detection — keypoints need finer updates
        lrf=0.01,           # Final LR = lr0 * lrf

        # --- Augmentation ---
        # NOTE: Pitch keypoints are sensitive to geometric augmentation.
        # Flipping is handled by flip_idx in the dataset config (left↔right symmetry).
        # We keep default augmentations which include mosaic, mixup, etc.

        # --- Hardware ---
        device=0,           # GPU 0
        amp=True,           # FP16 mixed precision (halves VRAM usage)
        workers=4,

        # --- Output ---
        project="runs/pose",
        name="football-pitch-keypoints",
        exist_ok=True,

        # --- Logging ---
        verbose=True,
    )

    # === RESULTS ===
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: runs/pose/football-pitch-keypoints/weights/best.pt")
    print(f"\nKey metrics:")
    print(f"  Box mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"  Pose mAP50: {results.results_dict.get('metrics/mAP50(P)', 'N/A'):.3f}")


if __name__ == "__main__":
    main()
