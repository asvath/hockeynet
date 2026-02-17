from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np

@dataclass
class JerseyIDDebouncer:
    """
    Debounce per-player team labels across frames.

    For each obj_id:
      - keep a stable jersey id
      - after lock_after number of frames, the jersey id will be locked for the obj_id (a player can't switch id suddenly)
      - if detected id for obj id is different after locking, it will be required to repeat `confirm_frames` times
        before reassignment.

    This kills jersey ID flickers while keeping real changes responsive.
    """
    mode: str = "parseq"  # "parseq" or "llm"
    init_frames: int = 20  # frames to collect before choosing initial stable ID via majority vote
    confirm_frames: int = 20
    lock_after: int = 30  # frames of agreement before locking in the ID
    locked_confirm_multiplier: int = 1  # once locked, require confirm_frames * this to override
    expire_after: Optional[int] = 300  # frames; set None to disable expiry, at 30fps: 10 seconds

    init_buffer: Dict[int, List[int]] = field(default_factory=dict)  # proposals collected before first stable ID
    stable_id: Dict[int, int] = field(default_factory=dict)
    stable_count: Dict[int, int] = field(default_factory=dict)  # how many times stable ID has been confirmed
    locked: Dict[int, bool] = field(default_factory=dict)
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
                self.init_buffer.pop(tid, None)
                self.stable_id.pop(tid, None)
                self.stable_count.pop(tid, None)
                self.locked.pop(tid, None)
                self.pending_id.pop(tid, None)
                self.pending_count.pop(tid, None)
                self.last_seen.pop(tid, None)

        smoothed = np.full(len(jersey_ids), -1, dtype=int) # initialize as -1, i.e. unknown

        for i, (tid, proposed) in enumerate(zip(out_obj_ids, jersey_ids)):
            tid = int(tid)

            proposed = int(proposed) if isinstance(proposed, str) else proposed

            # skip bad detections based on mode
            if self.mode == "llm":
                skip = proposed is None or jersey_confidence[i] != "high"
            else:
                skip = proposed == -1 or jersey_confidence[i] == -1

            if skip:
                smoothed[i] = self.stable_id.get(tid, -1)
                continue

            proposed = int(proposed)
            self.last_seen[tid] = self.frame_idx # last_seen[obj_id] = frame_idx, i.e the frame it was last seen

            # New track: collect proposals before committing to a stable ID
            if tid not in self.stable_id:
                buf = self.init_buffer.get(tid, [])
                buf.append(proposed)
                self.init_buffer[tid] = buf

                if len(buf) < self.init_frames:
                    smoothed[i] = -1  # not enough data yet
                    continue

                # majority vote from the buffer
                winner, winner_count = Counter(buf).most_common(1)[0]
                buf_len = len(buf)

                # # require >50% majority to commit
                # if winner_count <= buf_len / 2:
                #     if buf_len <= self.init_frames * 3: # buffer too small, keep collecting
                #         continue  # keep collecting

                if winner_count < self.init_frames:
                    continue

                self.stable_id[tid] = winner
                self.stable_count[tid] = winner_count
                self.locked[tid] = False
                self.pending_id[tid] = winner
                self.pending_count[tid] = 0
                self.init_buffer.pop(tid, None)
                smoothed[i] = winner
                continue

            stable = self.stable_id[tid]

            # If proposal agrees with stable, increment stable count and check for lock-in
            if proposed == stable:
                self.pending_id[tid] = stable
                self.pending_count[tid] = 0
                self.stable_count[tid] = self.stable_count.get(tid, 0) + 1

                # lock in the ID once we've seen enough agreement
                if not self.locked.get(tid) and self.stable_count[tid] >= self.lock_after:
                    self.locked[tid] = True

                smoothed[i] = stable
                continue

            # Proposal differs from stable: start/continue pending change
            if self.pending_id.get(tid) != proposed:
                self.pending_id[tid] = proposed
                self.pending_count[tid] = 1
            else:
                self.pending_count[tid] = self.pending_count.get(tid, 0) + 1

            # Once locked, require many more frames to override
            required = self.confirm_frames
            if self.locked.get(tid):
                required = self.confirm_frames * self.locked_confirm_multiplier

            # Commit change if it persists long enough
            if self.pending_count[tid] >= required:
                self.stable_id[tid] = proposed
                self.stable_count[tid] = 1  # reset: new ID starts fresh
                self.locked[tid] = False
                self.pending_id[tid] = proposed
                self.pending_count[tid] = 0
                smoothed[i] = proposed
            else:
                smoothed[i] = stable

        return smoothed

