"""Use supplemental detections only for existing, unmatched online tracks."""

import numpy as np

from src.world_model.m3w_camera_association import MotionAssociation
from src.world_model.m3w_streaming_boxes import match, overlap


class ContinuationBoxes:
    def __init__(self, require_previous_coarse):
        self.tracker = MotionAssociation(False)
        self.last_coarse = {}
        self.require_previous_coarse = require_previous_coarse

    def step(self, index, coarse, fine, motion=None):
        if (fine.ndim != 2 or fine.shape[1] != 6 or not np.isfinite(fine).all()
                or (fine[:, 2:4] <= fine[:, :2]).any()):
            raise ValueError('Expected finite supplemental xyxy, confidence, class')
        tracked = self.tracker.step(index, coarse, motion)
        active = self.tracker.active
        self.last_coarse = {k: self.last_coarse[k] for k in active if k in self.last_coarse}
        self.last_coarse.update({int(k): index for k in tracked[:, 0]})
        eligible = [k for k, history in active.items() if history[-1][0] < index
                    and (not self.require_previous_coarse or self.last_coarse[k] == index-1)]
        if not eligible or not len(fine):
            return tracked, np.empty((0, 7))
        # Coarse associations are already committed for this step. A fine box cannot
        # compete with one or create a new identity when the second match fails.
        fine = fine[~(overlap(fine[:, :4], coarse[:, :4]) > .5).any(axis=1)]
        prediction = np.stack([self.tracker.projected(active[k], index) for k in eligible])
        inverse = np.linalg.inv(self.tracker.pose)
        supplemental = []
        for i, j in match(prediction, fine):
            key, box = eligible[i], fine[j]
            active[key].append((index, box.copy(), inverse))
            supplemental.append((key, *box))
        return tracked, np.asarray(supplemental, dtype=float).reshape(-1, 7)
