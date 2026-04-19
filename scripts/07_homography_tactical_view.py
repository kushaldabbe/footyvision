"""
07_homography_tactical_view.py — Homography Transform + Top-Down Tactical View

=== WHAT THIS SCRIPT DOES ===
Combines all previous phases into a single pipeline:
  1. Detect players (YOLOv8s) + track (BoT-SORT) + classify teams (HSV/KMeans)
  2. Detect pitch keypoints (YOLOv8n-pose) [every N frames]
  3. Compute homography from detected keypoints → pitch template
  4. Transform player foot positions to top-down coordinates
  5. Render side-by-side: annotated broadcast view + 2D tactical radar

=== HOMOGRAPHY — THE CORE CONCEPT ===

A homography is a 3×3 matrix H that maps points from one plane to another.
Given a point (x, y) in the broadcast image, we compute where it lies on the
real pitch (X, Y) in centimeters:

    [X']     [h11 h12 h13] [x]
    [Y']  =  [h21 h22 h23] [y]
    [w']     [h31 h32 h33] [1]

    X = X'/w',  Y = Y'/w'   (perspective divide)

To compute H, we need ≥4 point correspondences between detected keypoints
(image pixels) and their known real-world positions (pitch template).

cv2.findHomography() uses RANSAC to robustly estimate H even if some
keypoint detections are noisy or incorrect.

=== WHY THIS WORKS FOR A FLAT SURFACE ===

Homography assumes all points lie on a SINGLE PLANE. A football pitch is flat,
so this assumption holds perfectly for foot positions (which touch the ground).

For head positions, it would be slightly wrong (players are ~1.8m tall, adding
vertical offset), but feet are on the ground plane → accurate transform.

=== AV/ROBOTICS PARALLEL — THIS IS THE BIG ONE ===

This is DIRECTLY what AV companies do:
  - Camera image → detect landmarks (lane markings, curbs, signs)
  - Match landmarks to HD Map (known world coordinates)
  - Compute camera-to-world transform
  - Project all detected objects into Bird's Eye View (BEV) for planning

Homography is the simplest case (flat ground, single plane).
AV systems extend this with:
  - IPM (Inverse Perspective Mapping) for flat ground with known camera intrinsics
  - Full 3D projection (PnP + depth) for non-planar scenes
  - Learned BEV transforms (Tesla Occupancy Networks, BEVFormer) that handle
    arbitrary 3D geometry

The companies you're targeting (Woven, Applied Intuition, Bosch) all work
with BEV representations. Being able to explain homography → IPM → learned BEV
in an interview is extremely valuable.

=== RUNNING PITCH KEYPOINT DETECTION EVERY N FRAMES ===

The pitch markings don't move — only the camera does. We don't need to re-detect
keypoints every frame. Detecting every 30 frames (1.2s at 25fps) saves compute
and the homography stays valid as long as the camera hasn't moved too much.

If the camera does a fast pan, the keypoints will be re-detected on the next
scheduled frame and the homography will update.

Usage:
    python scripts/07_homography_tactical_view.py
"""

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
import supervision as sv
from ultralytics import YOLO

# Import our pitch template
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from footyvision.homography.pitch_config import SoccerPitchConfig


# === CONFIGURATION ===
SAMPLE_FRAMES = 150          # Frames for team classification sampling
KEYPOINT_INTERVAL = 30       # Re-detect pitch keypoints every N frames
KEYPOINT_CONF_THRESH = 0.5   # Minimum confidence for a keypoint to be used
MIN_KEYPOINTS = 4            # Minimum keypoints needed for homography

# HSV histogram params (same as Phase 3)
H_BINS = 16
S_BINS = 8
HIST_SIZE = [H_BINS, S_BINS]
HIST_RANGES = [0, 180, 0, 256]

# YOLO classes
CLASS_BALL = 0
CLASS_GK = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3

# Team colors
TEAM_COLORS = {
    0: sv.Color.from_hex("#FF4444"),   # Team A — red
    1: sv.Color.from_hex("#4488FF"),   # Team B — blue
    2: sv.Color.from_hex("#FFFF44"),   # Referees — yellow
}
TEAM_LABELS = {0: "Team A", 1: "Team B", 2: "Referee"}

# Radar view colors (BGR for OpenCV drawing)
RADAR_TEAM_COLORS = {
    0: (68, 68, 255),    # Team A — red (BGR)
    1: (255, 136, 68),   # Team B — blue (BGR)
    2: (68, 255, 255),   # Referee — yellow (BGR)
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_torso_crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """Extract torso region (top 20%-60% of bbox) for jersey color."""
    x1, y1, x2, y2 = map(int, bbox)
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)
    crop_h, crop_w = y2 - y1, x2 - x1
    if crop_h < 40 or crop_w < 15:
        return None
    torso = frame[y1 + int(crop_h * 0.20):y1 + int(crop_h * 0.60), x1:x2]
    return torso if torso.size > 0 else None


def compute_hsv_histogram(crop: np.ndarray) -> np.ndarray:
    """2D HSV histogram (H×S), normalized."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, HIST_SIZE, HIST_RANGES)
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def get_dominant_class(votes: list[int]) -> int:
    """Most common class ID from per-frame votes."""
    counts = defaultdict(int)
    for v in votes:
        counts[v] += 1
    return max(counts, key=counts.get)


def get_foot_position(bbox: np.ndarray) -> np.ndarray:
    """
    Get the foot (bottom-center) position from a bounding box.

    We use bottom-center because that's where the player touches the ground plane.
    The homography maps ground-plane points accurately.
    Top of the bounding box (head) would be ~1.8m above ground → wrong position.
    """
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, y2], dtype=np.float32)


def compute_homography(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    pitch_config: SoccerPitchConfig,
    conf_threshold: float = KEYPOINT_CONF_THRESH,
) -> np.ndarray | None:
    """
    Compute homography from detected image keypoints to pitch template.

    Uses RANSAC to robustly handle noisy or incorrect keypoint detections.
    RANSAC (Random Sample Consensus):
      1. Randomly pick 4 keypoint pairs
      2. Compute homography from those 4
      3. Count how many OTHER keypoints agree (inliers)
      4. Repeat many times, keep the homography with most inliers

    This makes the estimation robust to outliers — even if 2 out of 10
    keypoints are wrong, RANSAC will find the correct homography from
    the other 8.

    Args:
        keypoints_xy: (32, 2) array of detected keypoint pixel positions
        keypoints_conf: (32,) array of confidence scores
        pitch_config: Pitch template with real-world coordinates
        conf_threshold: Minimum confidence to use a keypoint

    Returns:
        3×3 homography matrix, or None if insufficient keypoints
    """
    pitch_coords = pitch_config.vertices_array  # (32, 2)

    # Filter to confident, visible keypoints
    mask = keypoints_conf >= conf_threshold
    if mask.sum() < MIN_KEYPOINTS:
        return None

    src_pts = keypoints_xy[mask]   # Image pixel coordinates
    dst_pts = pitch_coords[mask]   # Real-world pitch coordinates (cm)

    # cv2.findHomography with RANSAC
    # Returns: H (3×3 matrix), mask (which points are inliers)
    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=50.0,  # pixels — max reprojection error for inlier
    )

    if H is None:
        return None

    n_inliers = inlier_mask.sum() if inlier_mask is not None else 0
    n_used = mask.sum()
    print(f"    Homography: {n_inliers}/{n_used} inliers from {int(mask.sum())} keypoints")

    return H


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography H to transform 2D points.

    points: (N, 2) array of (x, y) pixel positions
    H: 3×3 homography matrix
    Returns: (N, 2) array of transformed coordinates in pitch space (cm)

    The math:
      [X', Y', w'] = H @ [x, y, 1]
      X = X'/w',  Y = Y'/w'   (perspective divide — this is the non-linear part)
    """
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # cv2.perspectiveTransform expects (N, 1, 2) shape
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def draw_pitch_radar(
    pitch_config: SoccerPitchConfig,
    player_positions: list[tuple[float, float, int]],
    radar_size: tuple[int, int] = (600, 400),
) -> np.ndarray:
    """
    Draw a 2D top-down pitch with player dots.

    This is the "tactical radar" — a bird's-eye view showing where every
    player is on the pitch. Standard in sports analytics.

    Args:
        pitch_config: Pitch dimensions
        player_positions: List of (x_cm, y_cm, team_id) tuples
        radar_size: Output image size (width, height) in pixels

    Returns:
        BGR image of the radar view
    """
    w_px, h_px = radar_size
    margin = 30  # pixels of margin around pitch

    # Create dark green background
    radar = np.full((h_px, w_px, 3), (34, 80, 34), dtype=np.uint8)  # dark green

    # Scale factors: pitch coordinates (cm) → pixel coordinates
    pitch_w = pitch_config.length   # X dimension (cm)
    pitch_h = pitch_config.width    # Y dimension (cm)
    scale_x = (w_px - 2 * margin) / pitch_w
    scale_y = (h_px - 2 * margin) / pitch_h

    def to_pixel(x_cm: float, y_cm: float) -> tuple[int, int]:
        """Convert pitch coordinates (cm) to radar pixel coordinates."""
        px = int(margin + x_cm * scale_x)
        py = int(margin + y_cm * scale_y)
        return px, py

    # Draw pitch lines (white)
    line_color = (200, 200, 200)
    for i1, i2 in pitch_config.edges:
        p1 = pitch_config.vertices[i1 - 1]  # keypoints are 1-indexed
        p2 = pitch_config.vertices[i2 - 1]
        cv2.line(radar, to_pixel(*p1), to_pixel(*p2), line_color, 1, cv2.LINE_AA)

    # Draw centre circle (approximate with polygon)
    center = to_pixel(pitch_config.length / 2, pitch_config.width / 2)
    radius_px = int(pitch_config.centre_circle_radius * scale_x)
    cv2.circle(radar, center, radius_px, line_color, 1, cv2.LINE_AA)

    # Draw centre spot
    cv2.circle(radar, center, 3, line_color, -1, cv2.LINE_AA)

    # Draw penalty spots
    left_spot = to_pixel(pitch_config.penalty_spot_distance, pitch_config.width / 2)
    right_spot = to_pixel(pitch_config.length - pitch_config.penalty_spot_distance, pitch_config.width / 2)
    cv2.circle(radar, left_spot, 3, line_color, -1, cv2.LINE_AA)
    cv2.circle(radar, right_spot, 3, line_color, -1, cv2.LINE_AA)

    # Draw players
    for x_cm, y_cm, team_id in player_positions:
        # Clamp to pitch boundaries (some noise is expected)
        x_cm = np.clip(x_cm, 0, pitch_config.length)
        y_cm = np.clip(y_cm, 0, pitch_config.width)
        px, py = to_pixel(x_cm, y_cm)
        color = RADAR_TEAM_COLORS.get(team_id, (200, 200, 200))
        cv2.circle(radar, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(radar, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

    return radar


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # === PATHS ===
    player_model_path = Path("runs/detect/runs/detect/football-detect/weights/best.pt")
    pitch_model_path = Path("runs/pose/runs/pose/football-pitch-keypoints/weights/best.pt")
    video_path = Path("data/sample_videos/football_clip.mp4")
    output_path = Path("outputs/04_tactical_view_result.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # === LOAD MODELS ===
    player_model = YOLO(str(player_model_path))
    pitch_model = YOLO(str(pitch_model_path))
    pitch_config = SoccerPitchConfig()

    print(f"Player model: {player_model_path}")
    print(f"Pitch model: {pitch_model_path}")
    print(f"Pitch template: {pitch_config.length/100}m × {pitch_config.width/100}m, 32 keypoints")

    # =========================================================================
    # PASS 1: Build team classification model (same as Phase 3)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PASS 1: Team classification sampling ({SAMPLE_FRAMES} frames)")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    track_hists: dict[int, list[np.ndarray]] = defaultdict(list)
    track_class_votes: dict[int, list[int]] = defaultdict(list)
    frame_count = 0

    while cap.isOpened() and frame_count < SAMPLE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = player_model.track(
            frame, persist=True, tracker="configs/botsort_football.yaml",
            conf=0.3, iou=0.5, classes=[1, 2, 3], verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        if detections.tracker_id is None:
            continue

        for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            tid, cid = int(tid), int(cid)
            track_class_votes[tid].append(cid)
            if cid in (CLASS_PLAYER, CLASS_GK):
                torso = extract_torso_crop(frame, bbox)
                if torso is not None:
                    track_hists[tid].append(compute_hsv_histogram(torso))

    cap.release()

    # Build KMeans model (k=2, players only)
    player_features, player_tids = [], []
    track_dominant_class = {tid: get_dominant_class(votes) for tid, votes in track_class_votes.items()}
    for tid, hists in track_hists.items():
        if track_dominant_class.get(tid) == CLASS_PLAYER and len(hists) >= 3:
            player_features.append(np.mean(hists, axis=0))
            player_tids.append(tid)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(np.array(player_features))
    print(f"  Team model built from {len(player_features)} player tracks")

    # =========================================================================
    # PASS 2: Full video — detection + tracking + teams + homography + radar
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PASS 2: Full pipeline with tactical radar")
    print(f"{'='*60}")

    player_model.predictor = None  # Reset tracker

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output is wider: broadcast view + radar side by side
    radar_w, radar_h = 600, 400
    out_w = w + radar_w
    out_h = max(h, radar_h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    # Annotators
    team_palette = sv.ColorPalette([TEAM_COLORS[0], TEAM_COLORS[1], TEAM_COLORS[2]])
    ellipse_ann = sv.EllipseAnnotator(thickness=2, color=team_palette)
    label_ann = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, text_padding=5,
        text_position=sv.Position.BOTTOM_CENTER, color=team_palette,
    )

    # State
    frame_count = 0
    current_H = None  # Current homography matrix
    run_track_teams: dict[int, int] = {}
    run_track_hists: dict[int, list[np.ndarray]] = defaultdict(list)
    run_track_class_votes: dict[int, list[int]] = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --- Player detection + tracking ---
        player_results = player_model.track(
            frame, persist=True, tracker="configs/botsort_football.yaml",
            conf=0.3, iou=0.5, classes=[1, 2, 3], verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(player_results)

        # --- Pitch keypoint detection (every N frames) ---
        if frame_count % KEYPOINT_INTERVAL == 1 or current_H is None:
            pitch_results = pitch_model(frame, verbose=False)[0]

            if pitch_results.keypoints is not None and len(pitch_results.keypoints) > 0:
                # Get keypoints from the first (and typically only) detection
                kpts = pitch_results.keypoints[0]  # shape: (32, 3) — x, y, conf

                # Handle different ultralytics output formats
                if hasattr(kpts, 'data'):
                    kpt_data = kpts.data.cpu().numpy()
                else:
                    kpt_data = np.array(kpts)

                if kpt_data.ndim == 3:
                    kpt_data = kpt_data[0]  # Remove batch dimension if present

                if kpt_data.shape[0] >= 32:
                    kpt_xy = kpt_data[:32, :2]
                    kpt_conf = kpt_data[:32, 2]

                    H = compute_homography(kpt_xy, kpt_conf, pitch_config)
                    if H is not None:
                        current_H = H

        # --- Team classification (same logic as Phase 3) ---
        if detections.tracker_id is not None:
            # Sampling phase
            if frame_count <= SAMPLE_FRAMES:
                for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
                    tid_int, cid_int = int(tid), int(cid)
                    run_track_class_votes[tid_int].append(cid_int)
                    if cid_int in (CLASS_PLAYER, CLASS_GK):
                        torso = extract_torso_crop(frame, bbox)
                        if torso is not None:
                            run_track_hists[tid_int].append(compute_hsv_histogram(torso))

                if frame_count == SAMPLE_FRAMES:
                    for tid, votes in run_track_class_votes.items():
                        dom_class = get_dominant_class(votes)
                        if dom_class == CLASS_REFEREE:
                            run_track_teams[tid] = 2
                        elif tid in run_track_hists and len(run_track_hists[tid]) >= 3:
                            avg_hist = np.mean(run_track_hists[tid], axis=0)
                            run_track_teams[tid] = int(kmeans.predict(avg_hist.reshape(1, -1))[0])

            # Classify new tracks
            for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
                tid_int, cid_int = int(tid), int(cid)
                if tid_int in run_track_teams:
                    continue
                if cid_int == CLASS_REFEREE:
                    run_track_teams[tid_int] = 2
                else:
                    torso = extract_torso_crop(frame, bbox)
                    if torso is not None:
                        hist = compute_hsv_histogram(torso)
                        run_track_teams[tid_int] = int(kmeans.predict(hist.reshape(1, -1))[0])

        # --- Build annotated broadcast frame ---
        annotated = frame.copy()
        player_positions = []  # For radar: (x_cm, y_cm, team_id)

        if detections.tracker_id is not None:
            team_indices = []
            labels = []
            foot_points = []

            for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
                tid_int = int(tid)
                team = run_track_teams.get(tid_int, 2)
                team_indices.append(team)
                labels.append(f"#{tid_int} {TEAM_LABELS[team]}")
                foot_points.append(get_foot_position(bbox))

            # Team-colored annotations
            original_cids = detections.class_id.copy()
            detections.class_id = np.array(team_indices)
            annotated = ellipse_ann.annotate(scene=annotated, detections=detections)
            annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)
            detections.class_id = original_cids

            # --- Transform foot positions to pitch coordinates ---
            if current_H is not None and len(foot_points) > 0:
                foot_array = np.array(foot_points, dtype=np.float32)
                pitch_positions = transform_points(foot_array, current_H)

                for (px, py), team in zip(pitch_positions, team_indices):
                    # Sanity check: only include points roughly on the pitch
                    if -1000 < px < 13000 and -1000 < py < 8000:
                        player_positions.append((px, py, team))

        # --- Draw radar view ---
        radar = draw_pitch_radar(pitch_config, player_positions, radar_size=(radar_w, radar_h))

        # --- Compose output frame (broadcast + radar) ---
        output_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        output_frame[:h, :w] = annotated
        # Center the radar vertically on the right side
        y_offset = (out_h - radar_h) // 2
        output_frame[y_offset:y_offset + radar_h, w:w + radar_w] = radar

        # Add homography status text
        status = "HOMOGRAPHY: ACTIVE" if current_H is not None else "HOMOGRAPHY: WAITING..."
        cv2.putText(output_frame, status, (w + 10, y_offset - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Frame {frame_count}/{total_frames}",
                    (w + 10, y_offset + radar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        writer.write(output_frame)

        if frame_count % 100 == 0:
            n_on_pitch = len(player_positions)
            print(f"  Frame {frame_count}/{total_frames} — "
                  f"{len(detections)} detections, "
                  f"{n_on_pitch} mapped to pitch, "
                  f"H={'OK' if current_H is not None else 'NONE'}")

    cap.release()
    writer.release()

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Output saved to: {output_path}")
    print(f"Homography status: {'Computed' if current_H is not None else 'Never computed (insufficient keypoints)'}")


if __name__ == "__main__":
    main()
