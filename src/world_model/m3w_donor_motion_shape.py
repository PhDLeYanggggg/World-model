"""Transfer motion shape, not absolute location, from other sites."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.world_model.m3w_opendd_velocity_candidate import frames
from src.world_model.m3w_radius_matched_turn_menu import make_menu, peak


PREFIXES = (4, 5, 6)
K = 8
NULL_SEED = 403


def motion_frame(history):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1] < 4 or h.shape[2] != 2 or not np.isfinite(h).all():
        raise ValueError("Require complete causal histories")
    rot, scale = frames(h)
    vel = (np.diff(h, axis=1) @ rot)/scale[:, None, None]
    return vel.reshape(len(h), 2*(h.shape[1]-1)), rot, scale


@dataclass(frozen=True)
class Library:
    tree: cKDTree
    offsets: np.ndarray
    null_offsets: np.ndarray
    sites: tuple

    @classmethod
    def fit(cls, donors, held, prefix):
        if held in donors or not donors or prefix not in (*PREFIXES, 8):
            raise ValueError("Require a disjoint donor roster and registered prefix")
        x, y, null = [], [], []
        rng = np.random.default_rng(NULL_SEED + prefix)
        for site, donor in sorted(donors.items()):
            h = donor["h"]
            if prefix == 8:
                full = donor["valid"].all(1)
                h, f = h[full], donor["y"][full]
            else:
                f = h[:, prefix:] - h[:, prefix-1:prefix]
            p = h[:, :prefix]
            v, rot, scale = motion_frame(p)
            cv = (p[:, -1:] - p[:, -2:-1])*np.arange(1, f.shape[1]+1)[None, :, None]
            offsets = ((f-cv) @ rot)/scale[:, None, None]
            if not np.isfinite(offsets).all():
                raise ValueError(f"Nonfinite donor target: {site}")
            x.append(v)
            y.append(offsets)
            null.append(offsets[rng.permutation(len(offsets))])
        x = np.concatenate(x)
        if len(x) < K:
            raise ValueError("Insufficient donor support")
        return cls(cKDTree(x), np.concatenate(y), np.concatenate(null), tuple(sorted(donors)))

    def predict(self, history):
        x, rot, scale = motion_frame(history)
        distance, at = self.tree.query(x, k=K, workers=1)
        offsets = np.stack((self.offsets[at].mean(1), self.null_offsets[at].mean(1)), axis=1)
        return (offsets @ rot[:, None].swapaxes(-1, -2))*scale[:, None, None, None], distance[:, -1]


def match_radius(cv, anchor, offsets):
    r, raw = peak(anchor-cv), peak(offsets)
    factor = np.divide(r[:, None], raw, out=np.zeros_like(raw), where=raw > 1e-12)
    p = cv[:, None] + offsets*factor[..., None, None]
    p = np.where((raw <= 1e-12)[..., None, None], anchor[:, None], p)
    np.testing.assert_allclose(peak(p-cv[:, None]), np.broadcast_to(r[:, None], raw.shape), atol=1e-10)
    return p


def choose(history, amplitude, offsets):
    """Add one candidate per arm, using the same within-history comparison."""
    cv, _, old = make_menu(history, 12, amplitude)
    options = match_radius(cv, old[:, 0], offsets[8])
    scores = []
    for prefix in PREFIXES:
        h = history[:, :prefix]
        cv0, _, old0 = make_menu(h, 8-prefix, amplitude)
        add = match_radius(cv0, old0[:, 0], offsets[prefix])
        y = history[:, prefix:] - h[:, -1:]
        ref = np.linalg.norm(cv0-y, axis=-1).mean(1)
        err = np.linalg.norm(np.concatenate((old0, add), axis=1)-y[:, None], axis=-1).mean(2)
        scale = np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(.05)
        scores.append((ref[:, None]-err)/scale[:, None]/(8-prefix))
    mean = np.mean(scores, axis=0)
    row = np.arange(len(cv))
    best = mean[:, :old.shape[1]].argmax(1)
    anchor = old[row, best]
    use = mean[:, old.shape[1]:] > mean[row, best, None]
    chosen = np.where(use[..., None, None], options, anchor[:, None])
    return cv, anchor, options, chosen, use
