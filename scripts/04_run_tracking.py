"""
04_run_tracking.py — Multi-Object Tracking with BoT-SORT (Camera Motion Compensation)

=== WHAT THIS SCRIPT DOES ===
Takes detection results from our trained YOLOv8s and adds TRACKING — assigning
a persistent ID to each detected person across frames.

=== WHY WE SWITCHED FROM SUPERVISION's BYTETRACK TO ULTRALYTICS BoT-SORT ===

Attempt 1 (sv.ByteTrack, threshold=0.8): 94 unique IDs for ~25 objects.
Attempt 2 (sv.ByteTrack, threshold=0.5): 630 IDs! (threshold was inverted —
    it's a max COST threshold, not min IoU. Lowering = stricter = worse.)

Root cause: IoU-only tracking CANNOT handle camera panning. When the broadcast
camera pans left/right, ALL players shift 50+ pixels in one frame. IoU between
consecutive frames drops to ~0 for every track, causing mass track breakage.

Solution: BoT-SORT with GMC (Global Motion Compensation). Before matching:
1. Estimate camera motion using sparse optical flow (Lucas-Kanade on feature points)
2. Compute an affine transform that describes the global camera movement
3. Apply this transform to all predicted track positions
4. NOW compute IoU — since camera motion is compensated, IoU is meaningful again

This is like: "before comparing where I think the player is vs where I see them,
first account for the fact that the whole image shifted 60px to the right."

=== BYTETRACK vs BoT-SORT ===

  ByteTrack:  Kalman + Hungarian + two-stage association (high + low confidence)
  BoT-SORT:   Everything ByteTrack has, PLUS:
              - GMC (Global Motion Compensation) via optical flow / ORB / ECC
              - Optional ReID (appearance features for re-identification)

  For football with broadcast cameras: BoT-SORT's GMC is essential.

=== KALMAN FILTER, HUNGARIAN ALGORITHM, TRACK LIFECYCLE ===
(Same concepts as documented in the previous ByteTrack version — Kalman predicts
position, Hungarian finds optimal assignment, tracks go through
tentative → confirmed → lost → deleted lifecycle.)

=== AV/ROBOTICS PARALLEL ===
Camera motion compensation is critical in AV too:
- Ego-motion compensation: the self-driving car is moving, so ALL objects shift
  in the image. Must subtract ego-motion before tracking.
- BoT-SORT's GMC is a 2D version of what AV does in 3D with IMU + odometry.

Usage:
    python scripts/04_run_tracking.py
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def main():
    # === PATHS ===
    model_path = Path("runs/detect/runs/detect/football-detect/weights/best.pt")
    video_path = Path("data/sample_videos/football_clip.mp4")
    output_path = Path("outputs/02_tracking_result.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # === LOAD MODEL ===
    model = YOLO(str(model_path))
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")

    # === VIDEO INFO ===
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  FPS: {video_info.fps}")
    print(f"  Duration: {video_info.total_frames / video_info.fps:.1f}s")

    # === ANNOTATORS (using supervision — better visuals than ultralytics default) ===
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(
            ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
             "#F7DC6F", "#BB8FCE", "#85C1E9", "#F0B27A", "#AED6F1"]
        ),
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=int(video_info.fps * 2),  # 2 seconds of trail
        position=sv.Position.BOTTOM_CENTER,
    )

    # === PROCESS VIDEO ===
    # We use ultralytics model.track() instead of model() + separate tracker.
    # model.track() internally runs BoT-SORT (or ByteTrack) with:
    #   - Kalman filter for motion prediction
    #   - Hungarian algorithm for assignment
    #   - GMC (sparse optical flow) for camera motion compensation
    #   - Two-stage association (ByteTrack's high+low confidence trick)
    #
    # The tracker config is in the default botsort.yaml:
    #   tracker_type: botsort
    #   track_high_thresh: 0.5   (high-confidence detection threshold)
    #   track_low_thresh: 0.1    (low-confidence threshold for 2nd stage)
    #   new_track_thresh: 0.6    (confidence to create a new track)
    #   track_buffer: 30         (frames to keep lost tracks)
    #   match_thresh: 0.8        (matching distance threshold)
    #   gmc_method: sparseOptFlow (camera motion compensation method)
    #
    # We use persist=True so tracks persist across model.track() calls.

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_count = 0
    total_tracks = set()

    print(f"\nRunning detection + BoT-SORT tracking (with camera motion compensation)...")
    print(f"Output: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # model.track(): detection + tracking in one call
        # persist=True: keep track state across frames (essential!)
        # tracker="botsort.yaml": use BoT-SORT (default) with GMC
        # conf=0.3: minimum detection confidence
        # iou=0.5: NMS IoU threshold
        # classes=[1,2,3]: only track GK, player, referee (skip ball class 0)
        results = model.track(
            frame,
            persist=True,
            tracker="configs/botsort_football.yaml",
            conf=0.3,
            iou=0.5,
            classes=[1, 2, 3],
            verbose=False,
        )[0]

        # Convert to supervision Detections for annotation
        detections = sv.Detections.from_ultralytics(results)

        # Collect unique track IDs
        if detections.tracker_id is not None:
            total_tracks.update(detections.tracker_id.tolist())

        # Build labels
        if detections.tracker_id is not None:
            labels = [
                f"#{tid} {model.names[cid]}"
                for tid, cid in zip(detections.tracker_id, detections.class_id)
            ]
        else:
            labels = [model.names[cid] for cid in detections.class_id]

        # Annotate
        annotated = frame.copy()
        annotated = trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = ellipse_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        writer.write(annotated)

        if frame_count % 50 == 0:
            n_active = len(detections.tracker_id) if detections.tracker_id is not None else 0
            print(f"  Frame {frame_count}/{total_frames} — "
                  f"{n_active} active tracks, {len(total_tracks)} total unique IDs")

    cap.release()
    writer.release()

    print(f"\nDone!")
    print(f"Processed {frame_count} frames")
    print(f"Total unique tracks assigned: {len(total_tracks)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
