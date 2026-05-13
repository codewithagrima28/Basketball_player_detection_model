"""
heatmap.py
Generates court heatmaps showing where each player (or team) spent time.
Outputs PNG images overlaid on a basketball court diagram.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


# Court colours
COURT_BG        = (245, 213, 170)   # wooden floor colour (BGR)
COURT_LINE      = (255, 255, 255)
PAINT_COLOR     = (220, 180, 140)
HEATMAP_ALPHA   = 0.6


def draw_court(width: int = 800, height: int = 440) -> np.ndarray:
    """Draw a top-down basketball court diagram."""
    court = np.full((height, width, 3), COURT_BG, dtype=np.uint8)
    lw = 2   # line width

    # Outer boundary
    cv2.rectangle(court, (20, 20), (width - 20, height - 20), COURT_LINE, lw)

    # Half court line
    cv2.line(court, (width // 2, 20), (width // 2, height - 20), COURT_LINE, lw)

    # Centre circle
    cv2.circle(court, (width // 2, height // 2), 60, COURT_LINE, lw)
    cv2.circle(court, (width // 2, height // 2), 4, COURT_LINE, -1)

    # Left paint / key
    paint_w, paint_h = 120, 160
    paint_top = (height - paint_h) // 2
    cv2.rectangle(court, (20, paint_top), (20 + paint_w, paint_top + paint_h), COURT_LINE, lw)
    cv2.rectangle(court, (20, paint_top), (20 + paint_w, paint_top + paint_h), PAINT_COLOR, -1)
    cv2.rectangle(court, (20, paint_top), (20 + paint_w, paint_top + paint_h), COURT_LINE, lw)

    # Left free-throw circle
    cv2.ellipse(court, (20 + paint_w, height // 2), (60, 60), 0, -90, 90, COURT_LINE, lw)

    # Left basket
    cv2.circle(court, (20 + 20, height // 2), 15, COURT_LINE, lw)
    cv2.line(court, (20, height // 2 - 20), (20, height // 2 + 20), COURT_LINE, lw * 2)

    # Left 3-point arc
    cv2.ellipse(court, (20 + 20, height // 2), (170, 170), 0, -68, 68, COURT_LINE, lw)
    cv2.line(court, (20, height // 2 - 130), (20 + 90, height // 2 - 130), COURT_LINE, lw)
    cv2.line(court, (20, height // 2 + 130), (20 + 90, height // 2 + 130), COURT_LINE, lw)

    # Right paint / key (mirror)
    rx = width - 20 - paint_w
    cv2.rectangle(court, (rx, paint_top), (width - 20, paint_top + paint_h), PAINT_COLOR, -1)
    cv2.rectangle(court, (rx, paint_top), (width - 20, paint_top + paint_h), COURT_LINE, lw)

    # Right free-throw circle
    cv2.ellipse(court, (rx, height // 2), (60, 60), 0, 90, 270, COURT_LINE, lw)

    # Right basket
    cv2.circle(court, (width - 20 - 20, height // 2), 15, COURT_LINE, lw)
    cv2.line(court, (width - 20, height // 2 - 20), (width - 20, height // 2 + 20), COURT_LINE, lw * 2)

    # Right 3-point arc
    cv2.ellipse(court, (width - 40, height // 2), (170, 170), 0, 112, 248, COURT_LINE, lw)
    cv2.line(court, (width - 20, height // 2 - 130), (width - 110, height // 2 - 130), COURT_LINE, lw)
    cv2.line(court, (width - 20, height // 2 + 130), (width - 110, height // 2 + 130), COURT_LINE, lw)

    return court


def generate_heatmap(
    positions: list[tuple],
    output_path: str,
    title: str = "Player Heatmap",
    color_map: int = cv2.COLORMAP_JET,
    court_w: int = 800,
    court_h: int = 440,
    source_w: int = 1280,
    source_h: int = 720,
) -> np.ndarray:
    """
    Generate a heatmap image for a list of (x, y) positions.
    Positions are in source video pixel coordinates and get mapped to court size.

    Returns the heatmap image (numpy array).
    """
    court = draw_court(court_w, court_h)

    if not positions:
        cv2.putText(court, "No data", (court_w // 2 - 40, court_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.imwrite(output_path, court)
        return court

    # Build density map
    density = np.zeros((court_h, court_w), dtype=np.float32)

    for (cx, cy) in positions:
        # Scale source coords → court coords
        px = int(cx / source_w * court_w)
        py = int(cy / source_h * court_h)
        px = max(0, min(court_w - 1, px))
        py = max(0, min(court_h - 1, py))
        density[py, px] += 1.0

    # Smooth with Gaussian blur
    density = cv2.GaussianBlur(density, (51, 51), 0)

    # Normalise and apply colormap
    if density.max() > 0:
        density = density / density.max()

    heat_u8 = (density * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heat_u8, color_map)

    # Mask out zero areas (keep court visible)
    mask = (heat_u8 > 15).astype(np.float32)
    mask_3ch = np.stack([mask, mask, mask], axis=2)

    blended = (court.astype(np.float32) * (1 - mask_3ch * HEATMAP_ALPHA) +
               heatmap_colored.astype(np.float32) * mask_3ch * HEATMAP_ALPHA)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Title bar
    title_bar = np.full((50, court_w, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(title_bar, title, (15, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    final = np.vstack([title_bar, blended])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, final)
    return final


def generate_team_heatmap(
    all_positions: dict,        # {track_id: [(cx, cy), ...]}
    team_assignments: dict,     # {track_id: team_label}
    output_dir: str,
    source_w: int = 1280,
    source_h: int = 720,
):
    """Generate one heatmap per player and one combined heatmap per team."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-player heatmaps
    for track_id, positions in all_positions.items():
        team = team_assignments.get(track_id, "Unknown")
        path = str(out / f"player_{track_id}_heatmap.png")
        generate_heatmap(
            positions, path,
            title=f"Player #{track_id} ({team}) — Court Coverage",
            source_w=source_w, source_h=source_h,
        )

    # Per-team combined heatmaps
    team_positions: dict[str, list] = {}
    for track_id, positions in all_positions.items():
        team = team_assignments.get(track_id, "Unknown")
        if team not in team_positions:
            team_positions[team] = []
        team_positions[team].extend(positions)

    for team, positions in team_positions.items():
        safe_name = team.replace(" ", "_").lower()
        path = str(out / f"{safe_name}_heatmap.png")
        generate_heatmap(
            positions, path,
            title=f"{team} — Combined Court Coverage",
            color_map=cv2.COLORMAP_HOT,
            source_w=source_w, source_h=source_h,
        )

    print(f"[INFO] Heatmaps saved to: {out}/")
    return str(out)
