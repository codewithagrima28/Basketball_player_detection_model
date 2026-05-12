"""
Utility functions for drawing detections and saving output.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


# Color constants (BGR)
COLOR_BALL = (0, 165, 255)        # Orange for ball
COLOR_PLAYER_DEFAULT = (0, 255, 0) # Green fallback for player
COLOR_UNKNOWN = (180, 180, 180)   # Gray for unclassified

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes, labels, team colors and track IDs on frame."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox

        if det.class_name == "ball":
            color = COLOR_BALL
            label = f"Ball {det.confidence:.0%}"
            _draw_ball(frame, det.bbox, color, label)
        else:
            color = det.team_color if det.team_color else COLOR_PLAYER_DEFAULT
            team_label = det.team if det.team else "Player"
            id_label = f"#{det.track_id}" if det.track_id is not None else ""
            label = f"{team_label} {id_label} {det.confidence:.0%}".strip()
            _draw_player(frame, det.bbox, color, label)

    # Stats overlay
    _draw_stats_overlay(frame, detections)
    return frame


def _draw_player(frame: np.ndarray, bbox: tuple, color: tuple, label: str):
    """Draw a player bounding box with filled label bar."""
    x1, y1, x2, y2 = bbox

    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    label_y = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 2), color, -1)

    # Label text (white)
    cv2.putText(frame, label, (x1 + 3, label_y - 2),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def _draw_ball(frame: np.ndarray, bbox: tuple, color: tuple, label: str):
    """Draw ball as circle with label."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = max((x2 - x1), (y2 - y1)) // 2

    cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 3, color, -1)

    # Label
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    lx, ly = cx - tw // 2, cy - r - 6
    cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
    cv2.putText(frame, label, (lx, ly - 1),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def _draw_stats_overlay(frame: np.ndarray, detections: list):
    """Draw a semi-transparent stats panel in the top-left corner."""
    players = [d for d in detections if d.class_name == "player"]
    balls = [d for d in detections if d.class_name == "ball"]

    teams = {}
    for p in players:
        t = p.team or "Unknown"
        teams[t] = teams.get(t, 0) + 1

    lines = [
        f"Players: {len(players)}",
        f"Ball: {'Yes' if balls else 'No'}",
    ]
    for team, count in sorted(teams.items()):
        lines.append(f"  {team}: {count}")

    padding = 8
    line_height = 22
    panel_w = 160
    panel_h = padding * 2 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = 8 + padding + (i + 1) * line_height - 4
        cv2.putText(frame, line, (14, y),
                    FONT, 0.48, (230, 230, 230), 1, cv2.LINE_AA)


def save_output(frame: np.ndarray, path: str):
    """Save annotated frame as image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, frame)


def get_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create a VideoWriter for MP4 output."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def resize_frame(frame: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    """Resize frame keeping aspect ratio if larger than max_dim."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))
