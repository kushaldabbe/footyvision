"""
02_train_detector.py — Fine-tune YOLOv8s for Football Player Detection

=== WHAT THIS SCRIPT DOES ===
Fine-tunes a pretrained YOLOv8s model on the football-players-detection dataset.
"Fine-tuning" means we start from a model already trained on COCO (80 classes, 330k images)
and adapt it to our specific domain (4 classes: ball, goalkeeper, player, referee).

=== WHY FINE-TUNE INSTEAD OF TRAINING FROM SCRATCH? ===
The pretrained model already knows how to detect generic objects — edges, textures, shapes.
We only need to teach it what footballers/ball look like specifically. This is called
"transfer learning" and requires far less data and compute than training from zero.

Think of it like this: teaching someone who already speaks English to learn football
commentary is much easier than teaching a baby to speak AND learn football simultaneously.

=== KEY TRAINING PARAMETERS EXPLAINED ===
- epochs: Number of full passes through the training dataset. More = better (to a point).
  Too many → overfitting (model memorizes training data instead of generalizing).
- imgsz: Input image resolution. YOLO resizes all images to this square size.
  640 is the default. Higher = more detail but slower and more VRAM.
- batch: Number of images processed simultaneously. Limited by VRAM.
  GTX 1650Ti (4GB) → batch=8 is safe. Larger batches = smoother gradients.
- lr0: Initial learning rate. Controls how big the weight updates are.
  0.01 is the default. For fine-tuning, this is usually fine.
- patience: Early stopping — if validation loss doesn't improve for this many epochs, stop.
  Prevents overfitting and saves time.

=== AV/ROBOTICS PARALLEL ===
This is identical to what perception teams do: take a COCO/nuScenes pretrained detector
and fine-tune it on company-specific data (their sensor setup, target objects, environment).

Usage:
    python scripts/02_train_detector.py
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    # === PATHS ===
    # The data.yaml tells YOLO where the images and labels are, and what classes to detect.
    # It also defines the train/val/test splits.
    data_yaml = Path("data/datasets/football-players-detection/data.yaml")

    # Where to save the trained model
    output_dir = Path("models/trained")
    output_dir.mkdir(parents=True, exist_ok=True)

    # === MODEL SELECTION ===
    # YOLOv8 comes in 5 sizes: n (nano), s (small), m (medium), l (large), x (xlarge)
    #
    # | Model   | Params | mAP (COCO) | Speed (GPU) |
    # |---------|--------|------------|-------------|
    # | yolov8n | 3.2M   | 37.3       | ~1ms        |
    # | yolov8s | 11.2M  | 44.9       | ~1.2ms      |  ← Our choice
    # | yolov8m | 25.9M  | 50.2       | ~1.8ms      |
    # | yolov8l | 43.7M  | 52.9       | ~2.4ms      |
    # | yolov8x | 68.2M  | 53.9       | ~3.5ms      |
    #
    # We pick YOLOv8s — best accuracy/speed tradeoff for GTX 1650Ti.
    # "yolov8s.pt" = pretrained weights on COCO (auto-downloaded on first run).
    model = YOLO("yolov8s.pt")

    # === TRAINING ===
    # model.train() handles everything: data loading, augmentation, optimization, evaluation.
    #
    # Under the hood, each training step does:
    # 1. Load a batch of images + labels
    # 2. Apply augmentations (mosaic, HSV shift, flips, scaling)
    # 3. Forward pass: image → backbone (feature extraction) → neck (feature fusion) → head (predictions)
    # 4. Compute loss: CIoU (box) + BCE (class) + DFL (distribution focal loss for regression)
    # 5. Backward pass: compute gradients via backpropagation
    # 6. Update weights via SGD optimizer
    # 7. Repeat for all batches = 1 epoch
    results = model.train(
        data=str(data_yaml),

        # --- Core hyperparameters ---
        epochs=50,          # Full passes through training data. 50 is a good start for fine-tuning.
        imgsz=640,          # Input resolution (pixels). Standard for YOLOv8.
        batch=4,            # Images per batch. 4 is safe for 4GB VRAM. (8 caused OOM)
        patience=15,        # Stop early if no improvement for 15 epochs (prevents overfitting).

        # --- Optimizer ---
        # SGD (Stochastic Gradient Descent) with momentum is YOLO's default.
        # Adam is an alternative (faster convergence but sometimes worse generalization).
        optimizer="auto",   # Let YOLO pick (defaults to SGD for training, AdamW for small datasets)
        lr0=0.01,           # Initial learning rate
        lrf=0.01,           # Final LR as fraction of lr0 (lr decays via cosine schedule: 0.01 → 0.0001)

        # --- Augmentation ---
        # These augmentations make the model robust to variations it'll see in real videos.
        # mosaic=1.0,       # Mosaic augmentation: stitches 4 images into one (default: 1.0)
        # flipud=0.0,       # Vertical flip probability (default: 0.0 — football pitch has orientation)
        # fliplr=0.5,       # Horizontal flip probability (default: 0.5 — left/right is symmetric)
        # hsv_h=0.015,      # HSV hue augmentation (default: 0.015)
        # hsv_s=0.7,        # HSV saturation augmentation (default: 0.7)
        # hsv_v=0.4,        # HSV value/brightness augmentation (default: 0.4)

        # --- Device ---
        device=0,           # GPU index. 0 = first GPU. Use "cpu" to force CPU.
        amp=True,           # Automatic Mixed Precision (FP16) — halves VRAM usage, ~30% faster
        workers=2,          # Data loading workers. Lower = less RAM pressure. Default is 8.

        # --- Output ---
        project="runs/detect",   # Parent directory for results
        name="football-detect",  # Experiment name (creates runs/detect/football-detect/)
        exist_ok=True,           # Overwrite if experiment name already exists

        # --- Logging ---
        verbose=True,       # Print detailed training info
        plots=True,         # Generate training plots (loss curves, PR curves, confusion matrix)
    )

    # === WHAT HAPPENS AFTER TRAINING ===
    # Results are saved to: runs/detect/football-detect/
    # Key files:
    #   weights/best.pt   — best model (highest validation mAP) ← USE THIS
    #   weights/last.pt   — model from the final epoch
    #   results.csv        — per-epoch metrics
    #   confusion_matrix.png — which classes get confused
    #   P_curve.png, R_curve.png — precision/recall vs confidence
    #   results.png        — loss and mAP curves over training

    best_model_path = Path("runs/detect/football-detect/weights/best.pt")
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Check runs/detect/football-detect/ for training plots and metrics.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
