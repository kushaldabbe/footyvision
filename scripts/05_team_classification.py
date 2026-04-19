"""
05_team_classification.py — Team Classification via HSV Color Clustering (v2)

=== WHAT THIS SCRIPT DOES ===
Takes the detection + tracking pipeline and adds TEAM ASSIGNMENT — grouping
players into Team A and Team B using their jersey colors.

=== KEY DESIGN DECISION (v2 fix) ===
The detector already classifies referee (class 3) and goalkeeper (class 1).
Instead of blindly clustering ALL detections with k=3, we:
  1. SKIP referees from clustering — assign them directly via YOLO class
  2. Cluster ONLY regular players (class 2) with k=2
  3. Assign goalkeepers to the nearest team cluster
  4. LOCK team assignments — once a track is classified, it never changes

This fixes the v1 bugs where:
  - Referee colors contaminated clusters (referee counted as Team A)
  - Players switched teams mid-video (no locking)
  - k=3 gave unbalanced clusters (18/5/2 instead of clean team split)

=== APPROACH: HSV Color Histograms + KMeans ===

Step 1: Collect player crops from first N frames (sampling phase)
        - Only class 2 (players) contribute histograms for clustering
Step 2: Extract TORSO region (middle ~40% of crop height)
Step 3: Convert torso crop BGR → HSV color space
Step 4: Compute 2D histogram over Hue and Saturation channels
Step 5: Average histograms per track, run KMeans (k=2)
Step 6: Lock team assignments — once set, never changes
Step 7: Process full video with team-colored annotations

=== COLOR SPACES — WHY HSV OVER RGB ===

RGB: brightness entangled with color. Dark red [100,0,0] and bright red
     [255,0,0] are far apart in RGB but perceptually the same color.

HSV: H = hue (what color), S = saturation (how vivid), V = value (brightness).
     Same jersey in shadow vs sunlight → H and S stay similar, only V changes.
     We histogram over H and S, ignoring V — robust to lighting.

=== KMEANS CLUSTERING ===

Unsupervised: no labels needed. Discovers team groups from color data.
1. Initialize k centers (kmeans++ for smart init)
2. Assign each point to nearest center
3. Move centers to mean of assigned points
4. Repeat until convergence

=== AV/ROBOTICS PARALLEL ===
- Color-based classification: traffic light detection (red/yellow/green)
- Using detector class to gate downstream processing: standard in AV pipelines
  (e.g., only run lane-fitting on "lane marking" detections, not on "vehicle")
- Locking classifications to avoid flickering: common in AV state estimation
  (e.g., a traffic light doesn't flip red→green→red every frame)

Usage:
    python scripts/05_team_classification.py
"""

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
import supervision as sv
from ultralytics import YOLO


# === CONFIGURATION ===
SAMPLE_FRAMES = 150  # Frames to sample for building color model (6s at 25fps)

# HSV histogram bins
H_BINS = 16   # 180 / 16 ≈ 11° per bin
S_BINS = 8    # 256 / 8 = 32 per bin
HIST_SIZE = [H_BINS, S_BINS]
HIST_RANGES = [0, 180, 0, 256]  # [H_min, H_max, S_min, S_max]

# YOLO class IDs from our trained model
CLASS_BALL = 0
CLASS_GK = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3

# Team display colors and labels
TEAM_COLORS = {
    0: sv.Color.from_hex("#FF4444"),   # Team A — red
    1: sv.Color.from_hex("#4488FF"),   # Team B — blue
    2: sv.Color.from_hex("#FFFF44"),   # Referees — yellow
}

TEAM_LABELS = {
    0: "Team A",
    1: "Team B",
    2: "Referee",
}


def extract_torso_crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """
    Extract the torso region from a player bounding box.

    Takes middle ~40% of crop height:
    - Skip top 20% (head, hair — noisy colors)
    - Skip bottom 40% (legs, feet, grass — dominant green)
    - Keep middle 40% (torso — jersey, most discriminative)
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Clamp to frame boundaries
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    crop_h = y2 - y1
    crop_w = x2 - x1

    # Skip tiny crops (distant players, partial detections)
    if crop_h < 40 or crop_w < 15:
        return None

    torso_top = y1 + int(crop_h * 0.20)
    torso_bot = y1 + int(crop_h * 0.60)
    torso = frame[torso_top:torso_bot, x1:x2]

    if torso.size == 0:
        return None
    return torso


def compute_hsv_histogram(crop: np.ndarray) -> np.ndarray:
    """
    Compute a normalized 2D HSV histogram from a BGR crop.

    2D histogram captures JOINT distribution of hue + saturation.
    Normalized to sum to 1 — independent of crop size (probability distribution).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, HIST_SIZE, HIST_RANGES)
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def get_dominant_class(votes: list[int]) -> int:
    """Return the most common class ID from a list of per-frame class votes."""
    counts = defaultdict(int)
    for v in votes:
        counts[v] += 1
    return max(counts, key=counts.get)


def main():
    # === PATHS ===
    model_path = Path("runs/detect/runs/detect/football-detect/weights/best.pt")
    video_path = Path("data/sample_videos/football_clip.mp4")
    output_path = Path("outputs/03_team_classification_result.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")

    # =========================================================================
    # PASS 1: Sample player crops and build team color model
    # =========================================================================
    # We only cluster class 2 (players) with k=2.
    # Referees (class 3) get a fixed label. GKs (class 1) get nearest team.
    print(f"\n{'='*60}")
    print(f"PASS 1: Sampling player crops (first {SAMPLE_FRAMES} frames)")
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

        results = model.track(
            frame, persist=True, tracker="configs/botsort_football.yaml",
            conf=0.3, iou=0.5, classes=[1, 2, 3], verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        if detections.tracker_id is None:
            continue

        for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            tid = int(tid)
            cid = int(cid)
            track_class_votes[tid].append(cid)

            # Only collect histograms for players AND goalkeepers (need GK for assignment)
            if cid in (CLASS_PLAYER, CLASS_GK):
                torso = extract_torso_crop(frame, bbox)
                if torso is not None:
                    hist = compute_hsv_histogram(torso)
                    track_hists[tid].append(hist)

        if frame_count % 50 == 0:
            print(f"  Frame {frame_count}/{SAMPLE_FRAMES} — "
                  f"{sum(len(h) for h in track_hists.values())} samples, "
                  f"{len(track_hists)} tracks with histograms")

    cap.release()

    # === Determine dominant class per track via majority vote ===
    track_dominant_class: dict[int, int] = {}
    for tid, votes in track_class_votes.items():
        track_dominant_class[tid] = get_dominant_class(votes)

    n_players = sum(1 for c in track_dominant_class.values() if c == CLASS_PLAYER)
    n_gk = sum(1 for c in track_dominant_class.values() if c == CLASS_GK)
    n_ref = sum(1 for c in track_dominant_class.values() if c == CLASS_REFEREE)
    print(f"\n  Track class breakdown: {n_players} players, {n_gk} GKs, {n_ref} referees")

    # === Build feature matrix from PLAYERS ONLY (class 2) ===
    player_tids = []
    player_features = []
    for tid, hists in track_hists.items():
        if track_dominant_class.get(tid) == CLASS_PLAYER and len(hists) >= 3:
            player_features.append(np.mean(hists, axis=0))
            player_tids.append(tid)

    X = np.array(player_features)
    print(f"  Clustering {len(X)} player tracks into 2 teams (k=2)...")

    # === KMeans k=2: Team A vs Team B ===
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Build the team assignment dict (this will be used in pass 2 via kmeans.predict)
    team_counts = defaultdict(int)
    for tid, label in zip(player_tids, labels):
        team_counts[int(label)] += 1

    print(f"  Cluster 0: {team_counts[0]} player tracks")
    print(f"  Cluster 1: {team_counts[1]} player tracks")

    # =========================================================================
    # PASS 2: Process full video with team-colored annotations
    # =========================================================================
    print(f"\n{'='*60}")
    print("PASS 2: Full video with team classification")
    print(f"{'='*60}")

    # Reset tracker state for a clean second pass
    model.predictor = None

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Create annotators ONCE outside the loop (v1 bug: recreated every frame)
    team_palette = sv.ColorPalette([TEAM_COLORS[0], TEAM_COLORS[1], TEAM_COLORS[2]])
    ellipse_ann = sv.EllipseAnnotator(thickness=2, color=team_palette)
    trace_ann = sv.TraceAnnotator(
        thickness=2, trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER, color=team_palette,
    )
    label_ann = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, text_padding=5,
        text_position=sv.Position.BOTTOM_CENTER, color=team_palette,
    )

    frame_count = 0
    # Track teams for pass 2 — LOCKED once assigned
    run_track_teams: dict[int, int] = {}
    run_track_hists: dict[int, list[np.ndarray]] = defaultdict(list)
    run_track_class_votes: dict[int, list[int]] = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(
            frame, persist=True, tracker="configs/botsort_football.yaml",
            conf=0.3, iou=0.5, classes=[1, 2, 3], verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        if detections.tracker_id is None:
            writer.write(frame)
            continue

        # --- During sampling window: collect data per track ---
        if frame_count <= SAMPLE_FRAMES:
            for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
                tid_int, cid_int = int(tid), int(cid)
                run_track_class_votes[tid_int].append(cid_int)
                if cid_int in (CLASS_PLAYER, CLASS_GK):
                    torso = extract_torso_crop(frame, bbox)
                    if torso is not None:
                        run_track_hists[tid_int].append(compute_hsv_histogram(torso))

            # At end of sampling: classify all tracks
            if frame_count == SAMPLE_FRAMES:
                for tid, votes in run_track_class_votes.items():
                    dom_class = get_dominant_class(votes)

                    if dom_class == CLASS_REFEREE:
                        run_track_teams[tid] = 2  # Referee — fixed label
                    elif tid in run_track_hists and len(run_track_hists[tid]) >= 3:
                        avg_hist = np.mean(run_track_hists[tid], axis=0)
                        team = int(kmeans.predict(avg_hist.reshape(1, -1))[0])
                        run_track_teams[tid] = team  # Player or GK → nearest cluster

                n_a = sum(1 for t in run_track_teams.values() if t == 0)
                n_b = sum(1 for t in run_track_teams.values() if t == 1)
                n_r = sum(1 for t in run_track_teams.values() if t == 2)
                print(f"  Frame {frame_count}: Team A={n_a}, Team B={n_b}, Ref={n_r}")

        # --- Classify new tracks (post-sampling) ---
        # LOCKED: once a track has a team, it NEVER changes.
        for bbox, tid, cid in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            tid_int, cid_int = int(tid), int(cid)
            if tid_int in run_track_teams:
                continue  # Already assigned — LOCKED

            if cid_int == CLASS_REFEREE:
                run_track_teams[tid_int] = 2
            else:
                torso = extract_torso_crop(frame, bbox)
                if torso is not None:
                    hist = compute_hsv_histogram(torso)
                    team = int(kmeans.predict(hist.reshape(1, -1))[0])
                    run_track_teams[tid_int] = team

        # --- Build labels and set team colors ---
        team_indices = []
        labels = []
        for tid, cid in zip(detections.tracker_id, detections.class_id):
            tid_int = int(tid)
            team = run_track_teams.get(tid_int, 2)  # Default to referee if unknown
            team_indices.append(team)
            labels.append(f"#{tid_int} {TEAM_LABELS[team]}")

        # Override class_id with team index for supervision color palette lookup
        original_class_ids = detections.class_id.copy()
        detections.class_id = np.array(team_indices)

        # Annotate frame
        annotated = frame.copy()
        annotated = trace_ann.annotate(scene=annotated, detections=detections)
        annotated = ellipse_ann.annotate(scene=annotated, detections=detections)
        annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)

        # Restore original class_ids (so tracker isn't confused next frame)
        detections.class_id = original_class_ids

        writer.write(annotated)

        if frame_count % 100 == 0:
            n_a = sum(1 for t in team_indices if t == 0)
            n_b = sum(1 for t in team_indices if t == 1)
            n_r = sum(1 for t in team_indices if t == 2)
            print(f"  Frame {frame_count}/{total_frames} — A:{n_a} B:{n_b} Ref:{n_r}")

    cap.release()
    writer.release()

    # === SUMMARY ===
    final_counts = defaultdict(int)
    for tid, team in run_track_teams.items():
        final_counts[team] += 1

    print(f"\n{'='*60}")
    print("TEAM CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tracks classified: {len(run_track_teams)}")
    for team_id in sorted(final_counts.keys()):
        print(f"  {TEAM_LABELS[team_id]}: {final_counts[team_id]} tracks")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
