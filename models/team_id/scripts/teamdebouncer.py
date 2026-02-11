from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class TeamDebouncer:
    """
    Debounce per-player team labels across frames.

    For each obj_id:
      - keep a stable_team
      - if proposed team differs, require it to repeat `confirm_frames` times
        before switching stable_team

    This kills 1-frame flicker while keeping real changes responsive.
    """
    confirm_frames: int = 10
    expire_after: Optional[int] = 90  # frames; set None to disable expiry, #at 30fps: 3 seconds

    stable_team: Dict[int, int] = field(default_factory=dict)
    pending_team: Dict[int, int] = field(default_factory=dict)
    pending_count: Dict[int, int] = field(default_factory=dict)
    last_seen: Dict[int, int] = field(default_factory=dict)
    frame_idx: int = 0

    def update(self, out_obj_ids: np.ndarray, team_ids: np.ndarray) -> np.ndarray:
        """
        out_obj_ids: (N,) array of track IDs for the current frame
        team_ids:    (N,) array of per-ID team labels from k-means for the current frame
        returns:     (N,) smoothed team labels in the same order
        """
        out_obj_ids = np.asarray(out_obj_ids)
        team_ids = np.asarray(team_ids)

        if out_obj_ids.shape[0] != team_ids.shape[0]:
            raise ValueError("out_obj_ids and team_ids must have the same length")

        self.frame_idx += 1

        # Optional: expire stale IDs so memory doesn't grow forever
        if self.expire_after is not None:
            cutoff = self.frame_idx - self.expire_after # anything seen before frame cutoff is forgotten
            stale = [tid for tid, seen in self.last_seen.items() if seen < cutoff] # last_seen[tid] = frame_idx
            for tid in stale:
                self.stable_team.pop(tid, None)
                self.pending_team.pop(tid, None)
                self.pending_count.pop(tid, None)
                self.last_seen.pop(tid, None)

        smoothed = np.empty_like(team_ids, dtype=int)

        for i, (tid, proposed) in enumerate(zip(out_obj_ids, team_ids)):
            tid = int(tid)
            proposed = int(proposed)
            self.last_seen[tid] = self.frame_idx # last_seen[obj_id] = frame_idx, i.e the frame it was last seen

            # First time seeing this track: accept immediately
            if tid not in self.stable_team: # object is not in stable team,
                self.stable_team[tid] = proposed
                self.pending_team[tid] = proposed
                self.pending_count[tid] = 0
                smoothed[i] = proposed
                continue

            stable = self.stable_team[tid]

            # If proposal agrees with stable, clear pending
            if proposed == stable:
                self.pending_team[tid] = stable
                self.pending_count[tid] = 0
                smoothed[i] = stable
                continue

            # Proposal differs from stable: start/continue pending change
            # Keep pending until, we confirm to switch id (e.g. after 10 frames votes)
            if self.pending_team.get(tid) != proposed:
                self.pending_team[tid] = proposed
                self.pending_count[tid] = 1
            else:
                self.pending_count[tid] = self.pending_count.get(tid, 0) + 1

            # Commit change if it persists
            if self.pending_count[tid] >= self.confirm_frames:
                self.stable_team[tid] = proposed
                self.pending_team[tid] = proposed
                self.pending_count[tid] = 0
                smoothed[i] = proposed
            else:
                smoothed[i] = stable

        return smoothed
