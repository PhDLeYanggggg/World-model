from __future__ import annotations

from src.stage22_pipeline import _cluster_points, _heatmap


def test_stage22_cluster_points_empty():
    goals = _cluster_points(__import__("numpy").zeros((0, 2), dtype="float32"))
    assert goals == []


def test_stage22_heatmap_shape():
    import numpy as np

    h = _heatmap(np.array([[10.0, 10.0], [20.0, 20.0]], dtype="float32"), 100, 100, bins=16)
    assert h.shape == (16, 16)
    assert h.max() <= 1.0

