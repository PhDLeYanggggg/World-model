import numpy as np

from src import stage41_candidate_distiller as distill


def test_stage41_candidate_distiller_slice_metrics() -> None:
    labels = {
        "floor_fde": np.ones(4, dtype=float),
        "hard": np.array([True, False, True, False]),
        "easy": np.array([False, True, False, True]),
    }
    selected = np.array([0.5, 1.0, 0.75, 1.0])
    switch = np.array([True, False, True, False])
    out = distill._slice_metrics(selected, labels, switch, np.ones(4, dtype=bool))
    assert out["improvement"] == 0.1875
    assert out["hard_failure_improvement"] == 0.375
    assert out["easy_degradation"] == 0.0


def test_stage41_candidate_distiller_policy_grid_has_safe_no_switch() -> None:
    grid = distill._policy_grid()
    assert any(row["max_switch"] == 0.0 for row in grid)
    assert any(row["hard_only"] for row in grid)
