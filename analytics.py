"""
analytics.py
Tracks per-player statistics across a match:
  - Distance covered (in metres, estimated)
  - Speed (metres per second)
  - Court zone time (which area of court each player spent time in)
  - Ball possession estimates

All stats are stored per track ID and exported as a Pandas DataFrame.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Court zone definitions ────────────────────────────────────────────────
# Court is divided into 6 zones (normalised 0–1 coordinates):
#
#   |  Left Paint  |   Left Wing   |  Left Corner  |
#   |  Right Paint | Right Wing    | Right Corner  |
#
# Zones are assigned based on x position (left/right half)
# and y position (paint/wing/corner thirds)

COURT_ZONES = {
    "Left Paint":    (0.0, 0.0, 0.2, 1.0),   # (x1, y1, x2, y2) normalised
    "Left Wing":     (0.2, 0.0, 0.5, 0.33),
    "Left Corner":   (0.2, 0.67, 0.5, 1.0),
    "Mid Court":     (0.2, 0.33, 0.8, 0.67),
    "Right Wing":    (0.5, 0.0, 0.8, 0.33),
    "Right Corner":  (0.5, 0.67, 0.8, 1.0),
    "Right Paint":   (0.8, 0.0, 1.0, 1.0),
}

# Approximate metres per pixel conversion
# Standard NBA court: 28.65m x 15.24m
# We estimate based on frame size — can be calibrated per camera
COURT_LENGTH_M = 28.65
COURT_WIDTH_M  = 15.24


@dataclass
class PlayerStats:
    track_id: int
    team: str = "Unknown"
    frames_seen: int = 0
    total_distance_px: float = 0.0      # raw pixel distance
    total_distance_m: float = 0.0       # converted to metres
    max_speed_mps: float = 0.0          # metres per second
    current_speed_mps: float = 0.0
    zone_frames: dict = field(default_factory=lambda: defaultdict(int))
    positions: list = field(default_factory=list)   # (cx, cy) history
    last_position: Optional[tuple] = None


class PlayerAnalytics:
    """
    Collects and computes per-player stats throughout a match video.
    Call update() once per frame with current detections.
    Call get_summary() at the end to get a Pandas DataFrame.
    """

    def __init__(self, fps: float = 25.0, frame_w: int = 1280, frame_h: int = 720):
        self.fps = fps
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Pixels per metre (estimated from frame size vs real court size)
        self.px_per_m_x = frame_w / COURT_LENGTH_M
        self.px_per_m_y = frame_h / COURT_WIDTH_M

        self.players: dict[int, PlayerStats] = {}
        self.frame_count = 0

    def _get_centre(self, bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _px_to_metres(self, dx_px: float, dy_px: float) -> float:
        dx_m = dx_px / self.px_per_m_x
        dy_m = dy_px / self.px_per_m_y
        return float(np.sqrt(dx_m ** 2 + dy_m ** 2))

    def _get_zone(self, cx: float, cy: float) -> str:
        nx = cx / self.frame_w   # normalised x
        ny = cy / self.frame_h   # normalised y
        for zone, (x1, y1, x2, y2) in COURT_ZONES.items():
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return zone
        return "Mid Court"

    def update(self, detections: list):
        """
        Call once per frame. detections is a list of Detection objects
        from detector.py (must have track_id, bbox, team, class_name).
        """
        self.frame_count += 1

        for det in detections:
            if det.class_name != "player" or det.track_id is None:
                continue

            tid = det.track_id
            cx, cy = self._get_centre(det.bbox)

            if tid not in self.players:
                self.players[tid] = PlayerStats(
                    track_id=tid,
                    team=det.team or "Unknown",
                )

            p = self.players[tid]
            p.frames_seen += 1
            p.team = det.team or p.team
            p.positions.append((cx, cy))

            # Distance and speed
            if p.last_position is not None:
                dx = cx - p.last_position[0]
                dy = cy - p.last_position[1]
                dist_m = self._px_to_metres(abs(dx), abs(dy))
                p.total_distance_m += dist_m
                p.total_distance_px += np.sqrt(dx**2 + dy**2)

                speed_mps = dist_m * self.fps   # dist per frame × frames per sec
                p.current_speed_mps = speed_mps
                if speed_mps > p.max_speed_mps:
                    p.max_speed_mps = speed_mps

            p.last_position = (cx, cy)

            # Zone tracking
            zone = self._get_zone(cx, cy)
            p.zone_frames[zone] += 1

    def get_summary(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame with one row per player containing:
        track_id, team, time on court, distance covered, avg/max speed,
        most used zone, and zone breakdown percentages.
        """
        rows = []
        for tid, p in self.players.items():
            if p.frames_seen < 5:
                continue  # ignore ghost detections

            time_on_court_s = p.frames_seen / self.fps
            avg_speed = p.total_distance_m / time_on_court_s if time_on_court_s > 0 else 0

            # Zone percentages
            total_zone_frames = sum(p.zone_frames.values()) or 1
            zone_pcts = {
                zone: round(p.zone_frames.get(zone, 0) / total_zone_frames * 100, 1)
                for zone in COURT_ZONES
            }
            dominant_zone = max(p.zone_frames, key=p.zone_frames.get) if p.zone_frames else "Unknown"

            row = {
                "Player ID":          f"#{tid}",
                "Team":               p.team,
                "Time on Court (s)":  round(time_on_court_s, 1),
                "Distance (m)":       round(p.total_distance_m, 1),
                "Avg Speed (m/s)":    round(avg_speed, 2),
                "Max Speed (m/s)":    round(p.max_speed_mps, 2),
                "Dominant Zone":      dominant_zone,
                **{f"Zone: {z} (%)": zone_pcts[z] for z in COURT_ZONES},
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("Distance (m)", ascending=False).reset_index(drop=True)
        return df

    def get_heatmap_data(self, track_id: int) -> Optional[list]:
        """Returns list of (cx, cy) positions for a given player — used for heatmap."""
        if track_id not in self.players:
            return None
        return self.players[track_id].positions

    def get_all_positions(self) -> dict:
        """Returns {track_id: [(cx,cy), ...]} for all players."""
        return {tid: p.positions for tid, p in self.players.items()}

    def reset(self):
        self.players.clear()
        self.frame_count = 0
