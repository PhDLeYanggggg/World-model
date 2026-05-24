import numpy as np

from src import stage41_all_agent_risk_repair as repair


def test_stage41_all_agent_risk_repair_policy_blocks_easy() -> None:
    pred = {
        "candidate_score": np.array([[0.5, 0.2], [0.5, 0.1]], dtype=float),
        "endpoint_risk": np.array([0.1, 0.1], dtype=float),
        "harm": np.array([0.01, 0.01], dtype=float),
        "gain": np.array([0.9, 0.9], dtype=float),
        "physical": np.array([1.0, 1.0], dtype=float),
        "endpoint_delta": np.zeros((2, 2), dtype=float),
    }
    ds = {
        "floor_fde": np.array([1.0, 1.0], dtype=float),
        "hard": np.array([True, True]),
        "failure": np.array([True, True]),
        "easy": np.array([False, True]),
        "horizon": np.array([50, 50]),
        "current_xy": np.zeros((2, 2), dtype=float),
        "future_xy": np.zeros((2, 2), dtype=float),
        "normalizer": np.ones(2, dtype=float),
    }
    _selected, switch = repair._select_endpoint_risk_cap(
        pred,
        ds,
        {"endpoint_gain_min": 0.0, "endpoint_risk_max": 0.5, "gain_prob_min": 0.0, "harm_prob_max": 0.5, "max_switch": 1.0, "easy_block": True},
    )
    assert switch.tolist() == [True, False]


def test_stage41_all_agent_risk_repair_groups_cover_domain_horizon() -> None:
    ds = {"domain": np.array(["A", "A", "B"]), "horizon": np.array([10, 50, 50])}
    groups = repair._groups(ds, "domain_horizon")
    assert set(groups) == {"A:h10", "A:h50", "B:h10", "B:h50"}
    assert groups["A:h10"].sum() == 1
    assert groups["B:h50"].sum() == 1
