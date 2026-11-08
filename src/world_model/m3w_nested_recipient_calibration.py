"""Path-specific support ranking and strictly local calibrated action selection."""

import numpy as np
import torch
from torch import nn


THRESHOLDS = {
    "support": (-1., -.5, -.25, 0., .25, .5, .75, 1.),
    "gain": (-1., -.25, -.05, 0., .02, .05, .10, .25),
}


def roles(groups, held):
    ordered = sorted(groups)
    if len(ordered) != 9 or len(set(ordered)) != 9 or held not in ordered:
        raise ValueError("Require the original nine unique fitting sources")
    i = ordered.index(held)
    cal = [ordered[(i+j) % 9] for j in (1, 4)]
    return [g for g in ordered if g not in (*cal, held)], cal


class SupportRanker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 32),
                                  nn.SiLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.head(x)

    def score(self, x):
        good, bad, easy = self(x).sigmoid().unbind(1)
        return good-bad+.25*easy


def events(gain, base, known, easy):
    good = known & (base > 0.) & (gain > 0.)
    bad = (~known) | ((-gain) > .02*base)
    safe_easy = known & easy & ~bad
    return np.stack((good, bad, safe_easy), axis=1).astype(np.float32)


def pick(score, legal, inverse, count, threshold):
    """A quota computed from this query's past only; never fill with rejected rows."""
    n = len(score)
    accepted = legal & (score >= threshold)
    order = np.lexsort((np.arange(n), -score, ~accepted, inverse))
    sizes = np.bincount(inverse, minlength=len(count))
    rank = np.arange(n)-np.repeat(np.r_[0, np.cumsum(sizes)[:-1]], sizes)
    ids = order[(rank < count[inverse[order]]) & accepted[order]]
    on = np.zeros(n, bool)
    on[ids] = True
    return on


def select(cal_rows, grid):
    """Choose on calibration only; unsupported or empty recordings do not pass."""
    qualifies = [r for r in cal_rows if r["passes"] == r["views"] and r["switches"] > 0
                 and min(r["gain_both"]) > 0.]
    return max(qualifies, key=lambda r: (min(r["gain_both"]), -r["switches"],
                                         -grid.index(r["threshold"])))["threshold"] if qualifies else None
