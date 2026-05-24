from pathlib import Path

import numpy as np

from src import stage33_coordinate_invariant as s33


def test_stage33_paths_and_schema_are_isolated() -> None:
    assert s33.OUT_DIR == Path("outputs/stage33_coordinate_invariant")
    assert s33.DATA_DIR == Path("data/stage33_coordinate_invariant")
    assert "relative_speed" in s33.CI_FEATURE_NAMES
    assert "domain_external" in s33.CI_FEATURE_NAMES


def test_stage33_cluster_points_is_deterministic() -> None:
    pts = np.asarray([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    a = s33._cluster_points(pts, max_k=2)
    b = s33._cluster_points(pts, max_k=2)
    assert a.shape == (2, 2)
    assert np.allclose(a, b)


def test_stage33_standardize_to_ref_preserves_shape() -> None:
    x = np.eye(4)
    out = s33._standardize_to_ref(x, x, x * 2.0)
    assert out.shape == x.shape
    assert np.isfinite(out).all()
