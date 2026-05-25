import numpy as np

from src import stage41_fresh_all_agent_endpoint_specialist as specialist


def test_stage41_fresh_all_agent_policy_blocks_easy() -> None:
    pred = {
        "endpoint_delta": np.zeros((2, 2), dtype=float),
        "endpoint_risk": np.array([0.0, 0.0]),
        "candidate_score": np.array([[0.5, 0.1], [0.5, 0.1]], dtype=float),
        "gain": np.array([1.0, 1.0]),
        "harm": np.array([0.0, 0.0]),
        "failure": np.array([1.0, 1.0]),
        "physical": np.array([1.0, 1.0]),
    }
    labels = {
        "floor_fde": np.array([10.0, 10.0]),
        "candidate_fde": np.array([[10.0, 5.0], [10.0, 5.0]]),
        "current_xy": np.zeros((2, 2), dtype=float),
        "future_xy": np.zeros((2, 2), dtype=float),
        "normalizer": np.ones(2, dtype=float),
        "horizon": np.array([50, 50]),
        "hard": np.array([True, True]),
        "easy": np.array([False, True]),
        "failure": np.array([True, True]),
        "domain": np.array(["UCY", "UCY"]),
        "scene_id": np.array(["s", "s"]),
        "source_file": np.array(["a", "a"]),
    }
    policy = {
        "slices": {
            "UCY|50": {
                "candidate_gain_min": 0.0,
                "endpoint_gain_min": 0.0,
                "endpoint_risk_max": 1.0,
                "gain_prob_min": 0.0,
                "harm_prob_max": 1.0,
                "physical_prob_min": 0.0,
                "max_switch": 1.0,
                "easy_block": True,
            }
        }
    }
    selected, switch, _source = specialist._apply_policy(pred, labels, policy)
    assert selected.tolist() == [0.0, 10.0]
    assert switch.tolist() == [True, False]


def test_stage41_fresh_all_agent_score_rewards_t50() -> None:
    base = {"all_improvement": 0.0, "t50_improvement": 0.0, "t100_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}
    better = dict(base)
    better["t50_improvement"] = 0.1
    assert specialist._score(better) > specialist._score(base)
