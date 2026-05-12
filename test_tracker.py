"""
Unit tests for the PlayerTracker class.
Run with: pytest tests/test_tracker.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from tracker import PlayerTracker, iou
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockDetection:
    bbox: tuple
    class_name: str = "player"
    track_id: Optional[int] = None
    team: Optional[str] = None
    team_color: Optional[tuple] = None
    confidence: float = 0.9


# ── IoU tests ──────────────────────────────────────────────────────────────

def test_iou_identical_boxes():
    box = (0, 0, 100, 100)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = (0, 0, 50, 50)
    b = (100, 100, 200, 200)
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = (0, 0, 100, 100)
    b = (50, 0, 150, 100)
    result = iou(a, b)
    assert 0 < result < 1
    assert result == pytest.approx(50 * 100 / (100 * 100 + 100 * 100 - 50 * 100))


def test_iou_contained_box():
    outer = (0, 0, 100, 100)
    inner = (25, 25, 75, 75)
    result = iou(outer, inner)
    # inner area = 2500, outer area = 10000, union = 10000
    assert result == pytest.approx(2500 / 10000)


# ── Tracker tests ──────────────────────────────────────────────────────────

def test_tracker_assigns_ids_on_first_frame():
    tracker = PlayerTracker()
    dets = [MockDetection((0, 0, 50, 50)), MockDetection((100, 100, 150, 150))]
    ids = tracker.update(dets)
    assert len(ids) == 2
    assert ids[0] != ids[1]
    assert all(isinstance(i, int) for i in ids)


def test_tracker_consistent_ids_across_frames():
    tracker = PlayerTracker()
    dets1 = [MockDetection((0, 0, 50, 50))]
    ids1 = tracker.update(dets1)

    # Slight shift — should match the same track
    dets2 = [MockDetection((2, 2, 52, 52))]
    ids2 = tracker.update(dets2)

    assert ids1[0] == ids2[0]


def test_tracker_new_id_for_new_detection():
    tracker = PlayerTracker()
    dets1 = [MockDetection((0, 0, 50, 50))]
    ids1 = tracker.update(dets1)

    # Completely different location — should get a new ID
    dets2 = [MockDetection((0, 0, 50, 50)), MockDetection((400, 400, 450, 450))]
    ids2 = tracker.update(dets2)

    assert ids1[0] in ids2  # original ID reused
    assert len(set(ids2)) == 2  # two distinct IDs


def test_tracker_empty_detections():
    tracker = PlayerTracker()
    dets1 = [MockDetection((0, 0, 50, 50))]
    tracker.update(dets1)

    # No detections — should not crash, should return empty
    ids = tracker.update([])
    assert ids == []


def test_tracker_reset():
    tracker = PlayerTracker()
    dets = [MockDetection((0, 0, 50, 50))]
    tracker.update(dets)

    tracker.reset()
    ids_after = tracker.update(dets)
    assert ids_after[0] == 1  # IDs restart from 1


def test_tracker_track_ages_out():
    tracker = PlayerTracker(max_age=2)
    dets = [MockDetection((0, 0, 50, 50))]
    old_id = tracker.update(dets)[0]

    # Skip 3 frames without this detection
    tracker.update([])
    tracker.update([])
    tracker.update([])

    # Re-detect at same location — should get a NEW id (old one aged out)
    new_ids = tracker.update(dets)
    assert new_ids[0] != old_id
