from pathlib import Path

import numpy as np

from src import stage41_stratified_protocol as strat


def test_stage41_stratified_paths() -> None:
    assert strat.OUT_DIR == Path("outputs/stage41_stratified_protocol")
    assert strat.DATA_DIR == Path("data/stage41_stratified_protocol")


def test_stage41_stratified_split_mask_shape() -> None:
    idx = strat._candidate_split_index()
    n = int(idx["row_id"].max()) + 1
    mask = strat._split_mask("train", n)
    assert mask.dtype == bool
    assert mask.shape == (n,)
    assert np.any(mask)


def test_stage41_stratified_metric_aggregate() -> None:
    summary = strat._aggregate_metric([0.1, 0.2, 0.3])
    assert round(summary["mean"], 3) == 0.2
    assert summary["min"] == 0.1
    assert summary["max"] == 0.3


def test_stage41_tail_metric_score_rewards_low_tail_t50() -> None:
    weak_tail = {
        "all_improvement": 0.10,
        "t50_improvement": 0.12,
        "t100_improvement": 0.03,
        "hard_failure_improvement": 0.10,
        "easy_degradation": 0.0,
        "switch_rate": 0.08,
        "by_domain": {
            "ETH_UCY": {"t50_improvement": 0.00, "hard_failure_improvement": 0.05},
            "TrajNet": {"t50_improvement": 0.20, "hard_failure_improvement": 0.15},
        },
    }
    stronger_tail = dict(weak_tail)
    stronger_tail["by_domain"] = {
        "ETH_UCY": {"t50_improvement": 0.08, "hard_failure_improvement": 0.08},
        "TrajNet": {"t50_improvement": 0.18, "hard_failure_improvement": 0.14},
    }
    assert strat._metric_score(stronger_tail, "domain_tail") > strat._metric_score(weak_tail, "domain_tail")


def test_stage41_required_margins_are_not_t50_only() -> None:
    metrics = {
        "all_improvement": 0.01,
        "t50_improvement": 1.0,
        "hard_failure_improvement": 0.01,
        "easy_degradation": 0.0,
    }
    assert not strat._beats_stage37_required_margins(metrics)
