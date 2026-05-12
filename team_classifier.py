"""
Team Classifier
Classifies basketball players into teams based on jersey color
using K-Means clustering on HSV color histograms.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import Optional


# Pre-defined team color palette (BGR) with labels
TEAM_COLORS = {
    "Team A": (255, 50, 50),    # Blue-ish
    "Team B": (50, 50, 255),    # Red-ish
    "Unknown": (180, 180, 180), # Gray
}


class TeamClassifier:
    """
    Classifies players into teams by extracting dominant jersey colors.
    Uses K-Means on the upper-body crop to find the primary color,
    then assigns to the nearest team cluster.
    """

    def __init__(self, n_teams: int = 2, n_colors: int = 3):
        self.n_teams = n_teams
        self.n_colors = n_colors
        self.team_centroids: Optional[np.ndarray] = None
        self.team_labels: list[str] = []
        self.team_bgr_colors: list[tuple] = []
        self._sample_buffer: list[np.ndarray] = []
        self._fitted = False

    def _extract_jersey_region(self, crop: np.ndarray) -> np.ndarray:
        """Return the upper ~40% of the crop (torso/jersey area)."""
        h = crop.shape[0]
        return crop[int(h * 0.1): int(h * 0.5), :]

    def _dominant_color_hsv(self, region: np.ndarray) -> np.ndarray:
        """Extract dominant color of region in HSV using K-Means."""
        if region.size == 0:
            return np.array([0, 0, 128], dtype=np.float32)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Filter out very dark (shadows) and very light (white/grey) pixels
        mask = (pixels[:, 2] > 40) & (pixels[:, 1] > 30)
        filtered = pixels[mask]

        if len(filtered) < 10:
            filtered = pixels  # fallback

        k = min(self.n_colors, len(filtered))
        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        km.fit(filtered)

        # Pick the cluster with the most saturated dominant color
        counts = Counter(km.labels_)
        best = max(counts, key=lambda c: counts[c] * km.cluster_centers_[c][1])
        return km.cluster_centers_[best]

    def _hsv_to_bgr(self, hsv: np.ndarray) -> tuple:
        """Convert HSV array to BGR tuple."""
        px = np.uint8([[hsv]])
        bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def collect_sample(self, crop: np.ndarray):
        """
        Accumulate jersey color samples for auto-fitting team clusters.
        Call this on the first N frames before classify() is reliable.
        """
        region = self._extract_jersey_region(crop)
        color = self._dominant_color_hsv(region)
        self._sample_buffer.append(color)

        # Auto-fit once we have enough samples
        if len(self._sample_buffer) >= self.n_teams * 10 and not self._fitted:
            self._fit_teams()

    def _fit_teams(self):
        """Cluster accumulated color samples into N team centroids."""
        data = np.array(self._sample_buffer, dtype=np.float32)
        km = KMeans(n_clusters=self.n_teams, n_init=5, random_state=42)
        km.fit(data)

        self.team_centroids = km.cluster_centers_
        self.team_labels = [f"Team {chr(65 + i)}" for i in range(self.n_teams)]
        self.team_bgr_colors = [self._hsv_to_bgr(c) for c in self.team_centroids]
        self._fitted = True

    def classify(self, crop: np.ndarray) -> tuple[str, tuple]:
        """
        Classify a player crop into a team.
        Returns (team_label, bgr_color).
        """
        region = self._extract_jersey_region(crop)
        color_hsv = self._dominant_color_hsv(region)

        # Collect sample for auto-fitting
        self._sample_buffer.append(color_hsv)
        if len(self._sample_buffer) >= self.n_teams * 10 and not self._fitted:
            self._fit_teams()

        if not self._fitted:
            return "Unknown", TEAM_COLORS["Unknown"]

        # Find nearest centroid
        dists = np.linalg.norm(self.team_centroids - color_hsv, axis=1)
        idx = int(np.argmin(dists))
        return self.team_labels[idx], self.team_bgr_colors[idx]

    def reset(self):
        """Reset classifier state (e.g. for a new video)."""
        self._sample_buffer.clear()
        self._fitted = False
        self.team_centroids = None
        self.team_labels = []
        self.team_bgr_colors = []
