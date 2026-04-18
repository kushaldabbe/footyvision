"""
03_run_inference.py — Run trained YOLOv8s on sample football video

=== WHAT THIS SCRIPT DOES ===
Takes the best.pt model we just trained and runs it on the sample football video.
Uses the 'supervision' library to draw clean bounding boxes with class labels and
confidence scores on each frame, then saves the annotated video.

=== KEY CONCEPTS ===

1. Inference Pipeline:
   Video frame → resize to 640x640 → model forward pass → NMS → detections
   
2. Non-Maximum Suppression (NMS):
   The model outputs hundreds of overlapping candidate boxes per frame.
   NMS filters them: if two boxes overlap heavily (IoU > threshold), keep only
   the one with higher confidence. This is why you see `conf` and `iou` thresholds.

3. Supervision library:
   Instead of drawing boxes manually with OpenCV (tedious), supervision provides
   clean annotators: BoxAnnotator, LabelAnnotator, etc. It also handles video I/O.

=== AV/ROBOTICS PARALLEL ===
This is the inference/deployment step. In AV:
- Model runs on each camera frame in real-time
- Detections are passed to the tracking module (Phase 2)
- Same confidence/NMS thresholds need tuning per deployment scenario

Usage:
    python scripts/03_run_inference.py
"""

from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


def main():
    # === PATHS ===
    model_path = Path("runs/detect/runs/detect/football-detect/weights/best.pt")
    video_path = Path("data/sample_videos/football_clip.mp4")
    output_path = Path("outputs/01_detection_result.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # === LOAD MODEL ===
    # Loading the .pt file gives us the full model (architecture + trained weights).
    # The model remembers its class names, input size, etc.
    model = YOLO(str(model_path))
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")

    # === VIDEO INFO ===
    # Get video metadata before processing
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  FPS: {video_info.fps}")
    print(f"  Total frames: {video_info.total_frames}")
    print(f"  Duration: {video_info.total_frames / video_info.fps:.1f}s")

    # === ANNOTATORS ===
    # These define HOW to draw detections on frames.
    #
    # BoxAnnotator: draws bounding box rectangles
    # LabelAnnotator: draws text labels (class name + confidence)
    #
    # We use ellipse annotator instead of box — looks cleaner for sports
    # (draws an ellipse at the player's feet, common in football broadcasts)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5,
    )

    # === PROCESS VIDEO ===
    # sv.process_video runs a callback on each frame — clean pattern.
    # The callback receives a frame (numpy array) and returns the annotated frame.
    frame_count = 0

    def process_frame(frame, _) -> None:
        """Process a single video frame: detect → annotate → return."""
        nonlocal frame_count
        frame_count += 1

        # Run detection
        # conf: minimum confidence threshold (0.3 = 30%). Lower = more detections but more false positives.
        # iou: NMS IoU threshold. If two boxes overlap more than this, the weaker one is removed.
        # verbose=False: don't print per-frame results (too noisy for video)
        results = model(frame, conf=0.3, iou=0.5, verbose=False)[0]

        # Convert YOLO results → supervision Detections object
        # This is a universal format that works with all supervision annotators.
        detections = sv.Detections.from_ultralytics(results)

        # Create labels: "player 0.95", "referee 0.87", etc.
        labels = [
            f"{model.names[class_id]} {conf:.2f}"
            for class_id, conf
            in zip(detections.class_id, detections.confidence)
        ]

        # Draw annotations on the frame
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Progress indicator
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{video_info.total_frames} frames "
                  f"({frame_count/video_info.total_frames*100:.0f}%)")

        return annotated

    print(f"\nRunning inference on video...")
    print(f"Output will be saved to: {output_path}")

    sv.process_video(
        source_path=str(video_path),
        target_path=str(output_path),
        callback=process_frame,
    )

    print(f"\nDone! Annotated video saved to: {output_path}")
    print(f"Processed {frame_count} frames total.")


if __name__ == "__main__":
    main()
