from pathlib import Path

import numpy as np

from src import stage40_neural_optimization as s40


def test_stage40_paths_and_candidates() -> None:
    assert s40.OUT_DIR == Path("outputs/stage40_neural_optimization")
    assert s40.DATA_DIR == Path("data/stage40_neural_optimization")
    assert s40.CANDIDATE_NAMES[0] == "stage37_floor"
    assert len(s40.CANDIDATE_NAMES) >= 2


def test_stage40_candidate_arrays_have_stage37_floor() -> None:
    cand = s40._candidate_arrays("test")
    assert cand["candidate_delta"].ndim == 3
    assert cand["candidate_delta"].shape[1] == len(s40.CANDIDATE_NAMES)
    assert np.all(cand["stage37_idx"] == 0)


def test_stage40_metrics_fallback_zero_improvement() -> None:
    ds = s40._ds("test")
    fallback = ds["fallback_fde"].astype(float)
    m = s40._metrics(fallback.copy(), fallback, ds)
    assert m["all_improvement"] == 0.0
    assert m["easy_degradation"] == 0.0
