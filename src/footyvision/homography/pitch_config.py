"""
pitch_config.py — Standard Football Pitch Template (32 Keypoints)

Defines the real-world coordinates of 32 pitch landmarks in centimeters.
Used to compute homography transforms from broadcast camera view to top-down BEV.

Coordinate system:
  - Origin at top-left corner of the pitch
  - X axis: along the length (0 → 12000 cm = 120m)
  - Y axis: along the width  (0 → 7000 cm = 70m)

The 32 keypoints match the Roboflow football-field-detection-f07vi dataset.
Source: https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py

=== AV/ROBOTICS PARALLEL ===
This is a MAP — a known geometric template with fixed coordinates.
In AV, the equivalent is an HD Map (road geometry, lane markings, intersections).
Computing a homography from detected keypoints to this template is conceptually
identical to "localization" in AV — figuring out where the camera is relative
to a known world.

=== WHY 32 KEYPOINTS? ===
- Pitch corners (4): define the pitch boundary
- Penalty box corners (8): left and right
- Goal box corners (8): left and right
- Penalty spots (2): left and right
- Halfway line endpoints (2): top and bottom
- Centre circle points (4): top, bottom, left, right
- Penalty/goal box junction points (4): where boxes meet

Not all 32 are visible in any single broadcast frame — typically 8-15 are.
Homography only needs ≥4 point correspondences, so partial visibility is fine.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class SoccerPitchConfig:
    """Standard football pitch dimensions and keypoint coordinates."""

    # Pitch dimensions in centimeters (FIFA standard)
    width: int = 7000       # 70m
    length: int = 12000     # 120m

    # Box dimensions
    penalty_box_width: int = 4100    # 41m
    penalty_box_length: int = 2015   # 20.15m (from goal line)
    goal_box_width: int = 1832       # 18.32m
    goal_box_length: int = 550       # 5.5m
    centre_circle_radius: int = 915  # 9.15m
    penalty_spot_distance: int = 1100  # 11m from goal line

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """
        32 keypoint coordinates in (x, y) centimeters.
        Index 0 = keypoint #1, index 31 = keypoint #32.

        The numbering follows the Roboflow football-field-detection dataset.
        """
        w = self.width
        l = self.length
        pbw = self.penalty_box_width
        pbl = self.penalty_box_length
        gbw = self.goal_box_width
        gbl = self.goal_box_length
        ccr = self.centre_circle_radius
        psd = self.penalty_spot_distance

        return [
            # === LEFT SIDE (goal line at x=0) ===
            (0, 0),                             # 1:  top-left corner
            (0, (w - pbw) / 2),                 # 2:  left penalty box top
            (0, (w - gbw) / 2),                 # 3:  left goal box top
            (0, (w + gbw) / 2),                 # 4:  left goal box bottom
            (0, (w + pbw) / 2),                 # 5:  left penalty box bottom
            (0, w),                             # 6:  bottom-left corner

            (gbl, (w - gbw) / 2),              # 7:  left goal box right-top
            (gbl, (w + gbw) / 2),              # 8:  left goal box right-bottom

            (psd, w / 2),                       # 9:  left penalty spot

            (pbl, (w - pbw) / 2),              # 10: left penalty box right-top
            (pbl, (w - gbw) / 2),              # 11: left penalty-goal box junction top
            (pbl, (w + gbw) / 2),              # 12: left penalty-goal box junction bottom
            (pbl, (w + pbw) / 2),              # 13: left penalty box right-bottom

            # === MIDFIELD ===
            (l / 2, 0),                         # 14: halfway line top
            (l / 2, w / 2 - ccr),              # 15: centre circle top
            (l / 2, w / 2 + ccr),              # 16: centre circle bottom
            (l / 2, w),                         # 17: halfway line bottom

            # === RIGHT SIDE (goal line at x=length) ===
            (l - pbl, (w - pbw) / 2),         # 18: right penalty box left-top
            (l - pbl, (w - gbw) / 2),         # 19: right penalty-goal box junction top
            (l - pbl, (w + gbw) / 2),         # 20: right penalty-goal box junction bottom
            (l - pbl, (w + pbw) / 2),         # 21: right penalty box left-bottom

            (l - psd, w / 2),                  # 22: right penalty spot

            (l - gbl, (w - gbw) / 2),         # 23: right goal box left-top
            (l - gbl, (w + gbw) / 2),         # 24: right goal box left-bottom

            (l, 0),                             # 25: top-right corner
            (l, (w - pbw) / 2),                # 26: right penalty box top
            (l, (w - gbw) / 2),                # 27: right goal box top
            (l, (w + gbw) / 2),                # 28: right goal box bottom
            (l, (w + pbw) / 2),                # 29: right penalty box bottom
            (l, w),                             # 30: bottom-right corner

            # === CENTRE CIRCLE SIDES ===
            (l / 2 - ccr, w / 2),              # 31: centre circle left
            (l / 2 + ccr, w / 2),              # 32: centre circle right
        ]

    @property
    def vertices_array(self) -> np.ndarray:
        """Vertices as numpy array of shape (32, 2)."""
        return np.array(self.vertices, dtype=np.float32)

    # Edges connecting keypoints (for drawing the pitch outline)
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Left goal line
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        # Left goal box
        (7, 8),
        # Left penalty box
        (10, 11), (11, 12), (12, 13),
        # Halfway line
        (14, 15), (15, 16), (16, 17),
        # Right penalty box
        (18, 19), (19, 20), (20, 21),
        # Right goal box
        (23, 24),
        # Right goal line
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        # Touchlines (top and bottom)
        (1, 14), (14, 25),  # top touchline
        (6, 17), (17, 30),  # bottom touchline
        # Left box horizontal lines
        (2, 10), (3, 7), (4, 8), (5, 13),
        # Right box horizontal lines
        (18, 26), (23, 27), (24, 28), (21, 29),
    ])
