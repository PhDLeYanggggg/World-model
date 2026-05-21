from __future__ import annotations

import numpy as np


class Stage8MultiAgentInteractionEncoder:
    """Past-only multi-agent interaction feature encoder.

    This intentionally starts with scalar and kNN-style features. Stage 7/6
    showed graph attention did not help; Stage 8 treats graph features as an
    ablation, not as assumed progress.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def encode_scalar(self, history: np.ndarray) -> np.ndarray:
        last = history[-1]
        pos = last[:, 0:2]
        vel = last[:, 2:4]
        valid = np.linalg.norm(pos, axis=1) > 0
        pos = pos[valid]
        vel = vel[valid]
        if len(pos) < 2:
            return np.zeros(8, dtype=float)
        rel = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(rel, axis=2)
        dist[dist == 0] = np.inf
        nn = np.min(dist, axis=1)
        rel_v = vel[None, :, :] - vel[:, None, :]
        closing = -np.sum(rel * rel_v, axis=2) / np.maximum(dist, 1e-6)
        ttc = dist / np.maximum(closing, 1e-3)
        ttc[~np.isfinite(ttc)] = 999.0
        bbox = np.maximum(pos.max(axis=0) - pos.min(axis=0), 1.0)
        density = len(pos) / max(float(np.prod(bbox)), 1.0)
        return np.asarray(
            [
                min(float(np.min(nn)), 50.0) / 50.0,
                min(float(np.mean(nn)), 50.0) / 50.0,
                min(float(np.min(ttc)), 50.0) / 50.0,
                min(float(np.max(closing[np.isfinite(closing)])), 20.0) / 20.0,
                min(float(density), 1.0),
                min(float(len(pos)) / 64.0, 1.0),
                float(np.std(np.linalg.norm(vel, axis=1))),
                float(np.mean(np.linalg.norm(vel, axis=1))),
            ],
            dtype=float,
        )

    def encode_graph(self, history: np.ndarray) -> np.ndarray:
        scalar = self.encode_scalar(history)
        last = history[-1]
        pos = last[:, 0:2]
        valid = np.linalg.norm(pos, axis=1) > 0
        pos = pos[valid]
        if len(pos) < 2:
            return np.concatenate([scalar, np.zeros(4, dtype=float)])
        rel = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(rel, axis=2)
        dist[dist == 0] = np.inf
        knn = np.sort(dist, axis=1)[:, : min(self.k, len(pos) - 1)]
        graph = np.asarray(
            [
                min(float(np.mean(knn)), 50.0) / 50.0,
                min(float(np.std(knn)), 50.0) / 50.0,
                float(np.mean(knn < 2.0)),
                float(np.mean(knn < 5.0)),
            ],
            dtype=float,
        )
        return np.concatenate([scalar, graph])
