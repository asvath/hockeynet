from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
from scripts.utils.config import PROJECT_ROOT, PATHS

@dataclass
class JerseyIDDebouncer:
    """
    Debounce per-player team labels across frames.

    For each obj_id:
      - keep a stable_team
      - if proposed team differs, require it to repeat `confirm_frames` times
        before switching stable_team

    This kills 1-frame flicker while keeping real changes responsive.
    """
    confirm_frames: int = 10
    expire_after: Optional[int] = 300  # frames; set None to disable expiry, at 30fps: 10 seconds

    stable_id: Dict[int, int] = field(default_factory=dict)
    pending_id: Dict[int, int] = field(default_factory=dict)
    pending_count: Dict[int, int] = field(default_factory=dict)
    last_seen: Dict[int, int] = field(default_factory=dict)
    frame_idx: int = 0

    def update(self, out_obj_ids: np.ndarray, jersey_ids: np.ndarray, jersey_confidence:np.ndarray) -> np.ndarray:
        """
        out_obj_ids: (N,) array of object IDs for the current frame
        jersey_ids:  (N,) array of jersey-ID labels from llm's detection
        returns:     (N,) smoothed jersey labels in the same order
        """
        out_obj_ids = np.asarray(out_obj_ids)
        jersey_ids = np.asarray(jersey_ids)
        jersey_confidence = np.asarray(jersey_confidence)

        if out_obj_ids.shape[0] != jersey_ids.shape[0]:
            raise ValueError("out_obj_ids and jersey ids must have the same length")

        self.frame_idx += 1

        # Optional: expire stale IDs so memory doesn't grow forever
        if self.expire_after is not None:
            cutoff = self.frame_idx - self.expire_after # anything seen before frame cutoff is forgotten
            stale = [tid for tid, seen in self.last_seen.items() if seen < cutoff] # last_seen[tid] = frame_idx
            for tid in stale:
                self.stable_id.pop(tid, None)
                self.pending_id.pop(tid, None)
                self.pending_count.pop(tid, None)
                self.last_seen.pop(tid, None)

        smoothed = np.full(len(jersey_ids), -1, dtype=int) # initialize as -1, i.e. unknown

        for i, (tid, proposed) in enumerate(zip(out_obj_ids, jersey_ids)):
            tid = int(tid)

            # if llm outputs none or if llm's output is not confident, jersey id is set to stable id or -1 (i.e. unknown)
            if proposed is None or jersey_confidence[i] != "high":
                smoothed[i] = self.stable_id.get(tid, -1)
                continue

            proposed = int(proposed)
            self.last_seen[tid] = self.frame_idx # last_seen[obj_id] = frame_idx, i.e the frame it was last seen

            # First time seeing this track: accept immediately
            if tid not in self.stable_id: # object is not in stable id,
                self.stable_id[tid] = proposed
                self.pending_id[tid] = proposed
                self.pending_count[tid] = 0
                smoothed[i] = proposed
                continue

            stable = self.stable_id[tid]

            # If proposal agrees with stable, clear pending
            if proposed == stable:
                self.pending_id[tid] = stable
                self.pending_count[tid] = 0
                smoothed[i] = stable
                continue

            # Proposal differs from stable: start/continue pending change
            # Keep pending until, we confirm to switch id (e.g. after 10 frames votes)
            if self.pending_id.get(tid) != proposed:
                self.pending_id[tid] = proposed
                self.pending_count[tid] = 1
            else:
                self.pending_count[tid] = self.pending_count.get(tid, 0) + 1

            # Commit change if it persists
            if self.pending_count[tid] >= self.confirm_frames:
                self.stable_id[tid] = proposed
                self.pending_id[tid] = proposed
                self.pending_count[tid] = 0
                smoothed[i] = proposed
            else:
                smoothed[i] = stable

        return smoothed



