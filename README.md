# FootyVision ⚽

> Computer vision analytics pipeline for football — from broadcast footage to tactical insights.

## What It Does

FootyVision processes football match footage (broadcast or tactical camera) and produces:

- **Player detection & tracking** — detects players, goalkeepers, referees, and the ball with persistent IDs
- **Team classification** — automatically identifies which team each player belongs to
- **Tactical view** — transforms the broadcast perspective into a top-down 2D pitch view using homography
- **Analytics** — ball possession, player heatmaps, speed/distance stats, territory maps

## Architecture

```
Source Video
  ├── [YOLOv8s]      Player/GK/Referee Detection  ──►  [ByteTrack] Tracking
  │                                                          ├── Team Classification (HSV + KMeans)
  ├── [YOLOv8s-pose] Pitch Keypoint Detection (32 pts)  ──► Homography → Top-Down Transform
  ├── [YOLOv8n]      Ball Detection + Interpolation
  └── Scene Change Detection
         │
         ▼
  Outputs: Annotated Video | Tactical Radar | Heatmaps | Stats
```

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| YOLOv8 (ultralytics) | Object detection & pose estimation |
| ByteTrack (supervision) | Multi-object tracking |
| OpenCV | Image processing, homography, video I/O |
| scikit-learn | KMeans clustering for team classification |
| matplotlib | Heatmaps and visualizations |

## Hardware Target

Optimized to run on consumer hardware — developed on an **NVIDIA GTX 1650Ti (4GB VRAM)**.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd footyvision

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
footyvision/
├── data/                  # Datasets & sample videos (gitignored)
├── models/trained/        # Fine-tuned weights (gitignored)
├── notebooks/             # Jupyter exploration notebooks
├── src/footyvision/       # Main source package
│   ├── detection/         # YOLOv8 detection modules
│   ├── tracking/          # ByteTrack integration
│   ├── classification/    # Team classification
│   ├── homography/        # Perspective transforms
│   ├── visualization/     # Radar, heatmaps, overlays
│   └── utils/             # Common utilities
├── outputs/               # Generated results (gitignored)
├── documentation/         # Project plan, checklist, notes
├── requirements.txt
└── pyproject.toml
```