"""
Player Tracker
Lightweight IoU-based multi-object tracker for basketball players.
Assigns consistent track IDs across frames without heavy dependencies.
For production use, consider integrating ByteTrack or BoT-SORT via ultralytics.
"""

import numpy as np
from typing import Optional


def iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class Track:
    def __init__(self, track_id: int, bbox: tuple, max_age: int = 30):
        self.track_id = track_id
        self.bbox = bbox
        self.age = 0
        self.max_age = max_age
        self.hits = 1

    def update(self, bbox: tuple):
        self.bbox = bbox
        self.age = 0
        self.hits += 1

    def predict(self):
        """Age the track; returns False if it should be deleted."""
        self.age += 1
        return self.age <= self.max_age


class PlayerTracker:
    """
    Simple greedy IoU tracker.
    For each new set of detections, matches them to existing tracks
    using IoU and assigns consistent IDs.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list, frame: Optional[np.ndarray] = None) -> list[int]:
        """
        Update tracker with current frame detections.
        Returns list of track IDs in same order as detections.
        """
        if not detections:
            # Age all tracks
            self.tracks = [t for t in self.tracks if t.predict()]
            return []

        det_boxes = [d.bbox for d in detections]

        # Age existing tracks and remove dead ones
        active_tracks = []
        for t in self.tracks:
            if t.predict():
                active_tracks.append(t)
        self.tracks = active_tracks

        if not self.tracks:
            # No existing tracks — create new ones
            for bbox in det_boxes:
                self.tracks.append(Track(self._next_id, bbox, self.max_age))
                self._next_id += 1
            return [t.track_id for t in self.tracks[-len(det_boxes):]]

        # Build IoU cost matrix: tracks x detections
        cost_matrix = np.zeros((len(self.tracks), len(det_boxes)))
        for i, track in enumerate(self.tracks):
            for j, bbox in enumerate(det_boxes):
                cost_matrix[i, j] = iou(track.bbox, bbox)

        # Greedy matching (highest IoU first)
        matched_tracks = set()
        matched_dets = set()
        assignments = {}  # det_idx -> track

        while True:
            if cost_matrix.size == 0:
                break
            max_iou = cost_matrix.max()
            if max_iou < self.iou_threshold:
                break
            t_idx, d_idx = np.unravel_index(cost_matrix.argmax(), cost_matrix.shape)
            if t_idx in matched_tracks or d_idx in matched_dets:
                cost_matrix[t_idx, d_idx] = 0
                continue

            assignments[d_idx] = self.tracks[t_idx]
            self.tracks[t_idx].update(det_boxes[d_idx])
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            cost_matrix[t_idx, :] = 0
            cost_matrix[:, d_idx] = 0

        # Create new tracks for unmatched detections
        track_ids = []
        for d_idx, bbox in enumerate(det_boxes):
            if d_idx in assignments:
                track_ids.append(assignments[d_idx].track_id)
            else:
                new_track = Track(self._next_id, bbox, self.max_age)
                self.tracks.append(new_track)
                track_ids.append(self._next_id)
                self._next_id += 1

        return track_ids

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self._next_id = 1
