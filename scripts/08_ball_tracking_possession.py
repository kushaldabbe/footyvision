"""
08_ball_tracking_possession.py — Ball Detection, Interpolation & Possession

=== WHAT THIS SCRIPT DOES ===
Extends the full pipeline (script 07) to include ball tracking:
  1. Detect players + ball (all classes) → track players (BoT-SORT)
  2. Detect pitch keypoints → compute homography
  3. Track ball position (highest-confidence detection per frame)
  4. Interpolate missing ball positions (linear interpolation)
  5. Determine ball possession (nearest player within threshold)
  6. Render: annotated broadcast + tactical radar with ball + possession stats

=== BALL DETECTION CHALLENGE — SMALL OBJECT DETECTION ===

Our model's ball recall is only ~40% (confusion matrix: 60% missed as background).
This is a WELL-KNOWN problem in object detection:

  Why balls are hard to detect:
  - Tiny: ~15-30px in a 1920×1080 frame (< 0.1% of image area)
  - Fast-moving: motion blur makes features unrecognizable
  - Frequent occlusion: hidden by players' bodies, feet, or out of frame
  - Variable appearance: white ball on white lines, ball in shadow vs sunlight

  Industry solutions (not all used here, but know them for interviews):
  - SAHI (Sliced Aided Hyper Inference): tile image into overlapping crops,
    run detection on each crop, merge results. Effective but slow (4-16× more inference).
  - Higher resolution input: upscale or use larger input size (1280 vs 640)
  - Specialized architectures: smaller anchor sizes, FPN with more levels
  - Single-class model: train YOLOv8n on ball only → model focuses entirely on ball
  - Temporal context: use previous ball position to narrow search region

=== OUR APPROACH: DETECT + INTERPOLATE ===

Rather than training a better detector (diminishing returns on 4GB GPU),
we accept the 40% recall and USE INTERPOLATION to fill gaps:

  Frame 1:  ball detected at (500, 300)
  Frame 2:  ball NOT detected  → interpolate
  Frame 3:  ball NOT detected  → interpolate
  Frame 4:  ball detected at (560, 280)

  Interpolated positions: Frame 2 → (520, 293), Frame 3 → (540, 287)

This works because:
  - Ball moves smoothly (physics!) — linear interpolation is reasonable for short gaps
  - Gaps are usually 1-5 frames (ball is small, not invisible forever)
  - For longer gaps (>MAX_INTERP_GAP), we don't interpolate (ball likely out of frame)

=== BALL POSSESSION — SPATIAL PROXIMITY ===

Possession = which player is closest to the ball, in PITCH coordinates (not pixels).

Why pitch coordinates and not pixel coordinates?
  - In pixel space, a distant player near the camera looks "close" to the ball
  - In pitch coordinates (after homography transform), distances are real meters
  - Threshold: player within ~3m of ball = "possessing" the ball

This is directly analogous to AV scene understanding:
  - "Which lane is the ego vehicle in?" → proximity to lane center in world coordinates
  - "Which pedestrian is closest to the crosswalk?" → world-space distance
  - You NEVER compute these in pixel space in a real AV system

=== AV/ROBOTICS PARALLEL ===
- Small object detection: pedestrians far away, debris on road, traffic cones
- Temporal interpolation: sensor dropout (lidar returns nothing for 2 frames → predict)
- Object tracking with gaps: standard in AV (Kalman filter prediction fills gaps)
- Spatial proximity queries: "which lane?", "nearest vehicle ahead?", "gap to merge?"
  All computed in world coordinates (BEV), not image space.

Usage:
    python scripts/08_ball_tracking_possession.py
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

# Ball tracking params
BALL_CONF_THRESH = 0.15      # Lower threshold for ball (it's hard to detect!)
MAX_INTERP_GAP = 10          # Max frames to interpolate across (10 frames = 0.4s at 25fps)
POSSESSION_THRESH_CM = 350   # 3.5 meters — max distance for "possession" in cm
BALL_TRAIL_LENGTH = 30       # Show last 30 ball positions as trajectory trail

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
BALL_COLOR_BGR = (0, 255, 255)       # Yellow (BGR) for ball on broadcast
BALL_RADAR_COLOR = (255, 255, 255)   # White for ball on radar


# ============================================================================
# HELPER FUNCTIONS (reused from Phase 4)
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
    """Bottom-center of bbox = where player touches the ground plane."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, y2], dtype=np.float32)


def compute_homography(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    pitch_config: SoccerPitchConfig,
    conf_threshold: float = KEYPOINT_CONF_THRESH,
) -> np.ndarray | None:
    """Compute homography from detected image keypoints to pitch template using RANSAC."""
    pitch_coords = pitch_config.vertices_array
    mask = keypoints_conf >= conf_threshold
    if mask.sum() < MIN_KEYPOINTS:
        return None

    src_pts = keypoints_xy[mask]
    dst_pts = pitch_coords[mask]

    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=50.0,
    )
    if H is None:
        return None

    n_inliers = inlier_mask.sum() if inlier_mask is not None else 0
    print(f"    Homography: {n_inliers}/{int(mask.sum())} inliers")
    return H


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography H to transform 2D points from image to pitch coordinates."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


# ============================================================================
# BALL-SPECIFIC FUNCTIONS
# ============================================================================

def extract_ball_detection(detections: sv.Detections, conf_threshold: float = BALL_CONF_THRESH):
    """
    Pick the single best ball detection from a frame.

    Unlike players (many per frame), there's only ONE ball. We take the
    highest-confidence detection of class 0 (ball).

    Returns (x_center, y_center) in pixel coords, or None if no ball detected.
    """
    if detections.class_id is None or len(detections) == 0:
        return None

    ball_mask = detections.class_id == CLASS_BALL
    if not ball_mask.any():
        return None

    ball_bboxes = detections.xyxy[ball_mask]
    ball_confs = detections.confidence[ball_mask]

    # Filter by confidence threshold
    conf_mask = ball_confs >= conf_threshold
    if not conf_mask.any():
        return None

    # Pick highest confidence
    best_idx = ball_confs[conf_mask].argmax()
    bbox = ball_bboxes[conf_mask][best_idx]
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return np.array([center_x, center_y], dtype=np.float32)


def interpolate_ball_positions(
    raw_positions: dict[int, np.ndarray],
    total_frames: int,
    max_gap: int = MAX_INTERP_GAP,
) -> dict[int, np.ndarray]:
    """
    Fill gaps in ball detections using linear interpolation.

    This is the key technique for handling the 40% recall:
    - If ball is detected in frame 10 and frame 15, interpolate frames 11-14
    - If gap > max_gap frames, don't interpolate (ball likely out of frame / scene change)

    Linear interpolation: position(t) = start + (end - start) * (t - t_start) / (t_end - t_start)

    More sophisticated approaches (not used here, but know for interviews):
    - Kalman filter: predict next position using velocity estimate
    - Cubic spline: smoother curves for curved ball trajectories
    - Physics-based: model gravity + drag for aerial balls

    Args:
        raw_positions: {frame_number: np.array([x, y])} — detected positions only
        total_frames: Total number of frames in video
        max_gap: Maximum number of missing frames to interpolate across

    Returns:
        {frame_number: np.array([x, y])} — detected + interpolated positions
    """
    if len(raw_positions) < 2:
        return dict(raw_positions)

    interpolated = dict(raw_positions)
    sorted_frames = sorted(raw_positions.keys())

    for i in range(len(sorted_frames) - 1):
        f_start = sorted_frames[i]
        f_end = sorted_frames[i + 1]
        gap = f_end - f_start - 1

        if gap == 0 or gap > max_gap:
            continue  # No gap, or gap too large to interpolate

        pos_start = raw_positions[f_start]
        pos_end = raw_positions[f_end]

        # Linear interpolation for each missing frame
        for f in range(f_start + 1, f_end):
            t = (f - f_start) / (f_end - f_start)  # 0→1
            interpolated[f] = pos_start + t * (pos_end - pos_start)

    return interpolated


def determine_possession(
    ball_pos_cm: np.ndarray,
    player_positions: list[tuple[float, float, int]],
    threshold_cm: float = POSSESSION_THRESH_CM,
) -> int | None:
    """
    Determine which team has ball possession based on spatial proximity.

    In PITCH coordinates (centimeters), find the nearest player to the ball.
    If that player is within threshold_cm, their team "has possession".

    Args:
        ball_pos_cm: (x, y) ball position in pitch coordinates (cm)
        player_positions: List of (x_cm, y_cm, team_id)
        threshold_cm: Maximum distance for possession (default 3.5m = 350cm)

    Returns:
        Team ID (0 or 1) if a player is close enough, None otherwise
    """
    if len(player_positions) == 0:
        return None

    min_dist = float("inf")
    nearest_team = None

    for x, y, team_id in player_positions:
        if team_id == 2:  # Skip referees
            continue
        dist = np.sqrt((ball_pos_cm[0] - x) ** 2 + (ball_pos_cm[1] - y) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_team = team_id

    if min_dist <= threshold_cm:
        return nearest_team
    return None


# ============================================================================
# DRAWING FUNCTIONS
# ============================================================================

def draw_ball_marker(frame: np.ndarray, pos_px: np.ndarray, is_interpolated: bool = False):
    """Draw ball marker on broadcast frame. Dashed circle if interpolated."""
    cx, cy = int(pos_px[0]), int(pos_px[1])
    if is_interpolated:
        # Dotted ring for interpolated positions (less certain)
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            px = int(cx + 12 * np.cos(rad))
            py = int(cy + 12 * np.sin(rad))
            cv2.circle(frame, (px, py), 2, BALL_COLOR_BGR, -1)
    else:
        # Solid circle for detected positions
        cv2.circle(frame, (cx, cy), 12, BALL_COLOR_BGR, 2, cv2.LINE_AA)
    # Inner dot always
    cv2.circle(frame, (cx, cy), 4, BALL_COLOR_BGR, -1, cv2.LINE_AA)


def draw_ball_trail(frame: np.ndarray, trail: list[np.ndarray], is_radar: bool = False):
    """Draw ball trajectory trail (fading line connecting recent positions)."""
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        # Fade: older positions are more transparent (approximated by darker color)
        alpha = i / len(trail)
        if is_radar:
            color = tuple(int(c * alpha) for c in BALL_RADAR_COLOR)
        else:
            color = tuple(int(c * alpha) for c in BALL_COLOR_BGR)
        pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        pt2 = (int(trail[i][0]), int(trail[i][1]))
        thickness = max(1, int(2 * alpha))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_pitch_radar(
    pitch_config: SoccerPitchConfig,
    player_positions: list[tuple[float, float, int]],
    ball_pos_cm: np.ndarray | None = None,
    ball_trail_cm: list[np.ndarray] | None = None,
    possession_team: int | None = None,
    radar_size: tuple[int, int] = (600, 400),
) -> np.ndarray:
    """Draw 2D top-down pitch with players, ball, and trajectory."""
    w_px, h_px = radar_size
    margin = 30

    radar = np.full((h_px, w_px, 3), (34, 80, 34), dtype=np.uint8)

    pitch_w = pitch_config.length
    pitch_h = pitch_config.width
    scale_x = (w_px - 2 * margin) / pitch_w
    scale_y = (h_px - 2 * margin) / pitch_h

    def to_pixel(x_cm: float, y_cm: float) -> tuple[int, int]:
        px = int(margin + x_cm * scale_x)
        py = int(margin + y_cm * scale_y)
        return px, py

    # Draw pitch lines
    line_color = (200, 200, 200)
    for i1, i2 in pitch_config.edges:
        p1 = pitch_config.vertices[i1 - 1]
        p2 = pitch_config.vertices[i2 - 1]
        cv2.line(radar, to_pixel(*p1), to_pixel(*p2), line_color, 1, cv2.LINE_AA)

    # Centre circle + spots
    center = to_pixel(pitch_config.length / 2, pitch_config.width / 2)
    radius_px = int(pitch_config.centre_circle_radius * scale_x)
    cv2.circle(radar, center, radius_px, line_color, 1, cv2.LINE_AA)
    cv2.circle(radar, center, 3, line_color, -1, cv2.LINE_AA)
    left_spot = to_pixel(pitch_config.penalty_spot_distance, pitch_config.width / 2)
    right_spot = to_pixel(pitch_config.length - pitch_config.penalty_spot_distance, pitch_config.width / 2)
    cv2.circle(radar, left_spot, 3, line_color, -1, cv2.LINE_AA)
    cv2.circle(radar, right_spot, 3, line_color, -1, cv2.LINE_AA)

    # Draw ball trail on radar
    if ball_trail_cm is not None and len(ball_trail_cm) >= 2:
        trail_px = []
        for pos in ball_trail_cm:
            x_clamped = np.clip(pos[0], 0, pitch_config.length)
            y_clamped = np.clip(pos[1], 0, pitch_config.width)
            trail_px.append(np.array(to_pixel(x_clamped, y_clamped), dtype=np.float32))
        draw_ball_trail(radar, trail_px, is_radar=True)

    # Draw players
    for x_cm, y_cm, team_id in player_positions:
        x_cm = np.clip(x_cm, 0, pitch_config.length)
        y_cm = np.clip(y_cm, 0, pitch_config.width)
        px, py = to_pixel(x_cm, y_cm)
        color = RADAR_TEAM_COLORS.get(team_id, (200, 200, 200))
        cv2.circle(radar, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(radar, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw ball on radar
    if ball_pos_cm is not None:
        bx = np.clip(ball_pos_cm[0], 0, pitch_config.length)
        by = np.clip(ball_pos_cm[1], 0, pitch_config.width)
        bpx, bpy = to_pixel(bx, by)
        cv2.circle(radar, (bpx, bpy), 5, BALL_RADAR_COLOR, -1, cv2.LINE_AA)
        cv2.circle(radar, (bpx, bpy), 7, BALL_RADAR_COLOR, 1, cv2.LINE_AA)

    # Draw possession indicator
    if possession_team is not None:
        team_name = TEAM_LABELS.get(possession_team, "?")
        team_color = RADAR_TEAM_COLORS.get(possession_team, (200, 200, 200))
        cv2.putText(radar, f"Possession: {team_name}", (10, h_px - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, team_color, 1, cv2.LINE_AA)

    return radar


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # === PATHS ===
    player_model_path = Path("runs/detect/runs/detect/football-detect/weights/best.pt")
    pitch_model_path = Path("runs/pose/runs/pose/football-pitch-keypoints/weights/best.pt")
    video_path = Path("data/sample_videos/football_clip.mp4")
    output_path = Path("outputs/05_ball_possession_result.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # === LOAD MODELS ===
    player_model = YOLO(str(player_model_path))
    pitch_model = YOLO(str(pitch_model_path))
    pitch_config = SoccerPitchConfig()

    print(f"Player/ball model: {player_model_path}")
    print(f"  Classes: {player_model.names}")
    print(f"Pitch model: {pitch_model_path}")
    print(f"Pitch template: {pitch_config.length/100}m × {pitch_config.width/100}m")

    # =========================================================================
    # PASS 1: Team classification + raw ball detection
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PASS 1: Team classification + ball detection ({SAMPLE_FRAMES} frames)")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Team classification state (same as Phase 4)
    track_hists: dict[int, list[np.ndarray]] = defaultdict(list)
    track_class_votes: dict[int, list[int]] = defaultdict(list)

    # Ball raw detections: {frame_number: position_px}
    raw_ball_positions: dict[int, np.ndarray] = {}
    frame_count = 0

    while cap.isOpened() and frame_count < SAMPLE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Track players (exclude ball from tracker — ball tracking is manual)
        results = player_model.track(
            frame, persist=True, tracker="configs/botsort_football.yaml",
            conf=0.3, iou=0.5, classes=[1, 2, 3], verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if detections.tracker_id is not None:
            for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
                tid, cid = int(tid), int(cid)
                track_class_votes[tid].append(cid)
                if cid in (CLASS_PLAYER, CLASS_GK):
                    torso = extract_torso_crop(frame, bbox)
                    if torso is not None:
                        track_hists[tid].append(compute_hsv_histogram(torso))

        # Separate ball detection (explicitly filter to class 0 only)
        # NOTE: Must specify classes=[0] because .track() above sets a persistent
        # class filter on the model predictor that would otherwise exclude ball.
        ball_results = player_model(frame, conf=BALL_CONF_THRESH, iou=0.5, classes=[0], verbose=False)[0]
        ball_dets = sv.Detections.from_ultralytics(ball_results)
        ball_pos = extract_ball_detection(ball_dets)
        if ball_pos is not None:
            raw_ball_positions[frame_count] = ball_pos

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

    ball_det_rate = len(raw_ball_positions) / SAMPLE_FRAMES * 100
    print(f"  Team model built from {len(player_features)} player tracks")
    print(f"  Ball detected in {len(raw_ball_positions)}/{SAMPLE_FRAMES} sampling frames ({ball_det_rate:.0f}%)")

    # =========================================================================
    # PASS 2: Full ball detection (all frames)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PASS 2: Full ball detection scan ({total_frames} frames)")
    print(f"{'='*60}")

    # Reset and scan all frames for ball only (faster — no tracking overhead)
    # Reset predictor to clear any class filters from Pass 1 tracking
    player_model.predictor = None
    cap = cv2.VideoCapture(str(video_path))
    raw_ball_positions = {}  # Reset — we'll collect for ALL frames now
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        ball_results = player_model(frame, conf=BALL_CONF_THRESH, iou=0.5, classes=[0], verbose=False)[0]
        ball_dets = sv.Detections.from_ultralytics(ball_results)
        ball_pos = extract_ball_detection(ball_dets)
        if ball_pos is not None:
            raw_ball_positions[frame_count] = ball_pos

        if frame_count % 200 == 0:
            print(f"  Scanned {frame_count}/{total_frames} — "
                  f"ball found in {len(raw_ball_positions)} frames so far")

    cap.release()

    # Interpolate missing ball positions
    ball_det_rate = len(raw_ball_positions) / total_frames * 100
    print(f"\n  Raw ball detections: {len(raw_ball_positions)}/{total_frames} ({ball_det_rate:.1f}%)")

    ball_positions_px = interpolate_ball_positions(raw_ball_positions, total_frames)
    interp_count = len(ball_positions_px) - len(raw_ball_positions)
    coverage = len(ball_positions_px) / total_frames * 100
    print(f"  After interpolation: {len(ball_positions_px)}/{total_frames} ({coverage:.1f}%)")
    print(f"  Interpolated frames: {interp_count}")

    # =========================================================================
    # PASS 3: Full pipeline — render with ball + possession
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PASS 3: Full pipeline render with ball tracking & possession")
    print(f"{'='*60}")

    player_model.predictor = None  # Reset tracker state

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    current_H = None
    run_track_teams: dict[int, int] = {}
    run_track_hists: dict[int, list[np.ndarray]] = defaultdict(list)
    run_track_class_votes: dict[int, list[int]] = defaultdict(list)

    # Possession tracking
    possession_frames = {0: 0, 1: 0}  # team_id → frame count with possession
    ball_trail_px: list[np.ndarray] = []   # Recent ball positions in pixel coords
    ball_trail_cm: list[np.ndarray] = []   # Recent ball positions in pitch coords

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
                kpts = pitch_results.keypoints[0]
                if hasattr(kpts, 'data'):
                    kpt_data = kpts.data.cpu().numpy()
                else:
                    kpt_data = np.array(kpts)
                if kpt_data.ndim == 3:
                    kpt_data = kpt_data[0]
                if kpt_data.shape[0] >= 32:
                    kpt_xy = kpt_data[:32, :2]
                    kpt_conf = kpt_data[:32, 2]
                    H = compute_homography(kpt_xy, kpt_conf, pitch_config)
                    if H is not None:
                        current_H = H

        # --- Team classification ---
        if detections.tracker_id is not None:
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

        # --- Build annotated frame ---
        annotated = frame.copy()
        player_positions = []  # (x_cm, y_cm, team_id) for radar

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

            # Transform foot positions to pitch coordinates
            if current_H is not None and len(foot_points) > 0:
                foot_array = np.array(foot_points, dtype=np.float32)
                pitch_positions = transform_points(foot_array, current_H)
                for (px, py), team in zip(pitch_positions, team_indices):
                    if -1000 < px < 13000 and -1000 < py < 8000:
                        player_positions.append((px, py, team))

        # --- Ball position (from pre-computed + interpolated positions) ---
        ball_pos_px = ball_positions_px.get(frame_count)
        ball_pos_cm = None
        is_interpolated = frame_count not in raw_ball_positions
        possession_team = None

        if ball_pos_px is not None:
            # Draw ball on broadcast frame
            draw_ball_marker(annotated, ball_pos_px, is_interpolated=is_interpolated)

            # Update pixel trail
            ball_trail_px.append(ball_pos_px.copy())
            if len(ball_trail_px) > BALL_TRAIL_LENGTH:
                ball_trail_px.pop(0)
            draw_ball_trail(annotated, ball_trail_px)

            # Transform ball to pitch coordinates
            if current_H is not None:
                ball_cm = transform_points(ball_pos_px.reshape(1, 2), current_H)[0]
                if -1000 < ball_cm[0] < 13000 and -1000 < ball_cm[1] < 8000:
                    ball_pos_cm = ball_cm

                    # Update pitch trail
                    ball_trail_cm.append(ball_cm.copy())
                    if len(ball_trail_cm) > BALL_TRAIL_LENGTH:
                        ball_trail_cm.pop(0)

                    # Determine possession
                    possession_team = determine_possession(ball_cm, player_positions)
                    if possession_team is not None:
                        possession_frames[possession_team] += 1
        else:
            # No ball this frame — clear trails if gap is too long
            if len(ball_trail_px) > 0:
                ball_trail_px.clear()
                ball_trail_cm.clear()

        # --- Draw radar ---
        radar = draw_pitch_radar(
            pitch_config, player_positions,
            ball_pos_cm=ball_pos_cm,
            ball_trail_cm=ball_trail_cm if ball_pos_cm is not None else None,
            possession_team=possession_team,
            radar_size=(radar_w, radar_h),
        )

        # --- Compose output frame ---
        output_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        output_frame[:h, :w] = annotated
        y_offset = (out_h - radar_h) // 2
        output_frame[y_offset:y_offset + radar_h, w:w + radar_w] = radar

        # Status overlay
        status = "HOMOGRAPHY: ACTIVE" if current_H is not None else "HOMOGRAPHY: WAITING..."
        cv2.putText(output_frame, status, (w + 10, y_offset - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ball_status = "BALL: DETECTED" if ball_pos_px is not None and not is_interpolated else \
                      "BALL: INTERPOLATED" if ball_pos_px is not None else "BALL: LOST"
        ball_color = (0, 255, 0) if "DETECTED" in ball_status else \
                     (0, 255, 255) if "INTERPOLATED" in ball_status else (0, 0, 255)
        cv2.putText(output_frame, ball_status, (w + 10, y_offset - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 1)

        # Possession bar
        total_poss = possession_frames[0] + possession_frames[1]
        if total_poss > 0:
            pct_a = possession_frames[0] / total_poss * 100
            pct_b = possession_frames[1] / total_poss * 100
            poss_text = f"Possession: {TEAM_LABELS[0]} {pct_a:.0f}% | {TEAM_LABELS[1]} {pct_b:.0f}%"
            cv2.putText(output_frame, poss_text, (w + 10, y_offset + radar_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        cv2.putText(output_frame, f"Frame {frame_count}/{total_frames}",
                    (w + 10, y_offset + radar_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        writer.write(output_frame)

        if frame_count % 100 == 0:
            n_on_pitch = len(player_positions)
            print(f"  Frame {frame_count}/{total_frames} — "
                  f"{len(detections)} dets, {n_on_pitch} on pitch, "
                  f"ball={'yes' if ball_pos_px is not None else 'no'}, "
                  f"H={'OK' if current_H is not None else 'NONE'}")

    cap.release()
    writer.release()

    # =========================================================================
    # FINAL STATS
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 5 COMPLETE — Ball Detection & Possession")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"\nBall Detection Stats:")
    print(f"  Raw detections:   {len(raw_ball_positions)}/{total_frames} ({len(raw_ball_positions)/total_frames*100:.1f}%)")
    print(f"  After interp:     {len(ball_positions_px)}/{total_frames} ({len(ball_positions_px)/total_frames*100:.1f}%)")
    print(f"  Interpolated:     {interp_count} frames filled")
    print(f"  Max interp gap:   {MAX_INTERP_GAP} frames")
    print(f"  Conf threshold:   {BALL_CONF_THRESH}")

    total_poss = possession_frames[0] + possession_frames[1]
    if total_poss > 0:
        print(f"\nPossession Stats:")
        print(f"  {TEAM_LABELS[0]}: {possession_frames[0]/total_poss*100:.1f}% ({possession_frames[0]} frames)")
        print(f"  {TEAM_LABELS[1]}: {possession_frames[1]/total_poss*100:.1f}% ({possession_frames[1]} frames)")
        print(f"  Unassigned: {total_frames - total_poss} frames (ball lost or no player nearby)")
    else:
        print("\nPossession: Could not compute (no ball+player matches found)")

    print(f"\nHomography: {'Computed' if current_H is not None else 'Never computed'}")


if __name__ == "__main__":
    main()
