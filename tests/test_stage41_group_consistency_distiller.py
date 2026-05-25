import numpy as np

from src import stage41_group_consistency_distiller as gcd


def test_safe_float_replaces_nonfinite():
    x = np.asarray([0.1, np.inf, -np.inf, np.nan])
    out = gcd._safe_float(x, fill=7.0)
    assert np.allclose(out, [0.1, 7.0, 7.0, 7.0])


def test_policy_metrics_gates_proposal_switches_only(monkeypatch):
    data = {
        "proposal_switch": np.asarray([True, True, False]),
        "floor_ade": np.asarray([1.0, 1.0, 1.0]),
        "neural_ade": np.asarray([0.5, 2.0, 0.2]),
        "floor_xy": np.zeros((3, 4, 2), dtype=float),
        "neural_xy": np.zeros((3, 4, 2), dtype=float),
        "labels": {
            "horizon": np.asarray([50, 50, 50]),
            "hard": np.asarray([True, True, True]),
            "failure": np.asarray([False, False, False]),
            "easy": np.asarray([False, False, False]),
            "domain": np.asarray(["D", "D", "D"]),
            "candidate_fde": np.ones((3, 2)),
        },
        "keys": np.asarray(["a", "b", "c"], dtype=object),
    }

    def fake_eval(bundle, switch, name):
        selected = data["floor_ade"].copy()
        selected[switch] = data["neural_ade"][switch]
        return {
            "selected_metrics": {"all_improvement": 1.0 - selected.mean(), "t50_improvement": 1.0 - selected.mean(), "hard_failure_improvement": 1.0 - selected.mean(), "easy_degradation": 0.0},
            "multi_agent_metrics": {},
            "selected_stats": {},
            "floor_stats": {},
            "collision_delta_005": 0.0,
        }

    monkeypatch.setattr(gcd.jrc, "_evaluate_split_rollout", fake_eval)
    scores = {
        "safe_prob": np.asarray([0.9, 0.9, 0.9]),
        "gain_pred": np.asarray([0.5, 0.5, 0.5]),
        "unsafe_prob": np.asarray([0.1, 0.9, 0.1]),
    }
    metrics, switch = gcd._policy_metrics(scores, data, {"safe_min": 0.5, "gain_min": 0.0, "unsafe_max": 0.5})
    assert switch.tolist() == [True, False, False]
    assert metrics["all_improvement"] > 0.0


def test_fit_policy_returns_val_selected_policy(monkeypatch):
    data = {
        "proposal_switch": np.asarray([True, True, True, False]),
        "floor_ade": np.ones(4),
        "neural_ade": np.asarray([0.5, 0.4, 2.0, 0.1]),
        "hard": np.asarray([True, True, True, False]),
        "failure": np.asarray([False, False, False, False]),
        "easy": np.asarray([False, False, False, True]),
        "horizon": np.asarray([50, 50, 50, 50]),
        "domain": np.asarray(["D", "D", "D", "D"]),
        "candidate_fde": np.ones((4, 2)),
    }
    scores = {
        "safe_prob": np.asarray([0.9, 0.8, 0.2, 0.9]),
        "gain_pred": np.asarray([0.5, 0.4, -0.5, 0.8]),
        "unsafe_prob": np.asarray([0.1, 0.2, 0.9, 0.1]),
    }

    def fake_policy_metrics(scores_arg, data_arg, policy):
        switch = (
            data_arg["proposal_switch"]
            & (scores_arg["safe_prob"] >= policy["safe_min"])
            & (scores_arg["gain_pred"] >= policy["gain_min"])
            & (scores_arg["unsafe_prob"] <= policy["unsafe_max"])
        )
        selected = data_arg["floor_ade"].copy()
        selected[switch] = data_arg["neural_ade"][switch]
        imp = 1.0 - float(selected.mean())
        return {"all_improvement": imp, "t50_improvement": imp, "hard_failure_improvement": imp, "easy_degradation": 0.0, "collision_delta_vs_floor_005": 0.0}, switch

    monkeypatch.setattr(gcd, "_policy_metrics", fake_policy_metrics)
    policy, candidates = gcd._fit_policy(scores, data)
    assert policy["type"] == "group_consistency_distiller"
    assert candidates
    assert "safe_min" in policy
