import numpy as np

from src import stage41_endpoint_to_full_statistical_evidence as evidence


def test_bootstrap_positive_when_selected_is_better():
    floor = np.ones(128, dtype=np.float64)
    selected = floor * 0.9
    out = evidence._bootstrap(selected, floor, np.ones(128, dtype=bool), seed=7)
    assert out["low"] > 0.05
    assert out["n"] == 128


def test_bootstrap_bundle_has_ade_and_fde_slices():
    labels = {
        "horizon": np.asarray([10, 25, 50, 100] * 32, dtype=np.int16),
        "hard": np.asarray([True, False, True, False] * 32),
        "failure": np.asarray([False, False, True, False] * 32),
    }
    n = len(labels["horizon"])
    ev = {
        "selected_ade": np.ones(n) * 0.8,
        "floor_ade": np.ones(n),
        "selected_fde": np.ones(n) * 0.75,
        "floor_fde": np.ones(n),
        "multi": np.asarray([True, False] * (n // 2)),
    }
    out = evidence._bootstrap_bundle(ev, labels)
    assert out["ade"]["t50"]["low"] > 0
    assert out["fde"]["multi_agent"]["low"] > 0
