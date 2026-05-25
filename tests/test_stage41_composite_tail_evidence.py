import numpy as np

from src import stage41_composite_tail_evidence as evidence


def test_bootstrap_delta_positive_when_selected_beats_reference():
    selected = np.full(100, 8.0)
    reference = np.full(100, 9.0)
    floor = np.full(100, 10.0)
    mask = np.ones(100, dtype=bool)
    ci = evidence._bootstrap_delta(selected, reference, floor, mask, n=100, seed=7)
    assert ci["low"] > 0
    assert ci["bootstrap_n"] == 100


def test_slice_mask_supports_domain_t50():
    ds = {
        "horizon": np.asarray([50, 25, 50]),
        "hard": np.asarray([False, False, True]),
        "failure": np.asarray([False, True, False]),
        "domain": np.asarray(["UCY", "UCY", "ETH_UCY"]),
    }
    mask = evidence._slice_mask(ds, "domain_t50:UCY")
    assert np.array_equal(mask, np.asarray([True, False, False]))


def test_evidence_pass_requires_t100_ci():
    metrics = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "t100_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "collision_delta_vs_floor_005": 0.0,
        "by_domain": {"A": {"all_improvement": 0.1}, "B": {"t50_improvement": 0.1}},
    }
    bootstrap = {
        "all": {"low": 0.01},
        "t50": {"low": 0.01},
        "t100": {"low": 0.0},
        "hard_failure": {"low": 0.01},
    }
    assert not evidence._evidence_passes(metrics, bootstrap)
    bootstrap["t100"]["low"] = 0.01
    assert evidence._evidence_passes(metrics, bootstrap)
