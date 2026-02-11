from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from models.team_id.scripts.team_features import extract_all_colour_features
from scripts.team_id.team_features import canonicalize_teams


@dataclass
class TeamAssigner:
    """
    Wraps cluster_frame() with:
      - prototype learning on good frames (median team hue)
      - prototype fallback on bad frames (N<2 or sep low, cluster_frame returns all zeros)
    """
    sep_thresh: float = 5.0
    n_clusters: int = 2

    proto_history: int = 25        # rolling window length (how many good-frame medians to remember)
    proto_ready_min: int = 5       # need at least this many updates before trusting prototypes

    team0_hues: deque = field(init=False)
    team1_hues: deque = field(init=False)

    def __post_init__(self):
        self.team0_hues = deque(maxlen=self.proto_history)
        self.team1_hues = deque(maxlen=self.proto_history)

    def _proto0(self) -> Optional[float]:
        if len(self.team0_hues) < self.proto_ready_min:
            return None
        return float(np.median(np.array(self.team0_hues, dtype=float)))

    def _proto1(self) -> Optional[float]:
        if len(self.team1_hues) < self.proto_ready_min:
            return None
        return float(np.median(np.array(self.team1_hues, dtype=float)))

    @staticmethod
    def _circ_hue_dist(h: float, p: float) -> float:
        """Circular hue distance for OpenCV HSV where H in [0,179]."""
        dh = abs(float(h) - float(p))
        return min(dh, 180.0 - dh)

    def _update_protos_from_good_frame(self, labels: np.ndarray, feats: np.ndarray) -> None:
        """Update rolling hue prototypes from a good frame."""
        H = feats[:, 0]

        h0 = H[labels == 0]
        h1 = H[labels == 1]

        # Only update if both clusters actually exist
        if h0.size == 0 or h1.size == 0:
            return

        self.team0_hues.append(float(np.median(h0)))
        self.team1_hues.append(float(np.median(h1)))

    def _assign_by_protos(self, feats: np.ndarray) -> np.ndarray:
        """Assign each player to nearest team hue prototype."""
        p0 = self._proto0()
        p1 = self._proto1()

        N = feats.shape[0]
        if p0 is None or p1 is None:
            return np.zeros(N, dtype=int)  # not ready yet

        out = np.zeros(N, dtype=int)
        for i in range(N):
            h = feats[i][0]  # player hue
            # compare player hue to team hue
            d0 = self._circ_hue_dist(h, p0)
            d1 = self._circ_hue_dist(h, p1)
            # assign to the nearest team hue
            out[i] = 0 if d0 <= d1 else 1
        return out

    @staticmethod
    def cluster_frame(frame: np.ndarray, masks: np.ndarray, n_clusters, sep_thresh):
        """
        Run full team-clustering pipeline on a single frame.
        :param frame: (H, W, 3) BGR image
        :param masks: (N, H, W) decoded boolean masks
        :param n_clusters: number of teams
        :param sep_thresh: (optional) threshold for separating teams
        :return: canonicalized labels (N,)
        """
        # colour features from full masks
        feats = extract_all_colour_features(frame, masks)

        # only 1 or no players
        N = masks.shape[0]
        if N < n_clusters:
            return np.zeros(N, dtype=int), feats, 0.0

        # k-means clustering into two teams
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(feats)

        centers = kmeans.cluster_centers_
        sep = np.linalg.norm(centers[0] - centers[1])

        # only one team visible, assign all to 0, can be fixed with temporal smoothing later
        if sep < sep_thresh:
            return np.zeros(N, dtype=int), feats, 0

        # canonicalize so labels are consistent across frames
        labels = canonicalize_teams(labels, feats)

        return labels, feats, sep


    def step(self, frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Returns:
          labels: (N,) proposed team labels for this frame (k-means if good, else prototype fallback)
          feats:  (N,3) colour features (median HSV per player)
        """
        labels, feats, sep = self.cluster_frame(frame, masks, n_clusters=self.n_clusters, sep_thresh=self.sep_thresh)
        N = masks.shape[0]

        # more than equal to two players and clusters are clearly separated
        good_frame = (N >= self.n_clusters) and (sep >= self.sep_thresh)

        if good_frame:
            # labels should already be canonicalized inside your cluster_frame
            self._update_protos_from_good_frame(labels, feats)
            return labels.astype(int)

        # bad frame: override "all zeros" using prototypes (if ready)
        labels_fb = self._assign_by_protos(feats)
        return labels_fb.astype(int)
