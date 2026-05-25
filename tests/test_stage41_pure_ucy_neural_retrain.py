import numpy as np

from src import stage41_pure_ucy_neural_retrain as pure


def test_pure_ucy_split_uses_only_official_ucy_sources():
    assert pure._pure_ucy_split("/tmp/datasets/UCY/students01/students001-trajnet.txt") == "train"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/students03/obsmat.txt") == "train"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/zara01/obsmat.txt") == "val"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/zara02/obsmat.txt") == "test"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/zara03/crowds_zara03.txt") == "test"
    assert pure._pure_ucy_split("/tmp/datasets/TrajNet/Train/crowds/crowds_zara03.txt") == "unused"


def test_select_policy_falls_back_when_switch_budget_zero():
    pred = {
        "candidate_score": np.asarray([[1.0, 0.1], [1.0, 0.2]], dtype=np.float32),
        "gain": np.asarray([1.0, 1.0], dtype=np.float32),
        "harm": np.asarray([0.0, 0.0], dtype=np.float32),
        "physical": np.asarray([1.0, 1.0], dtype=np.float32),
    }
    ds = {
        "candidate_fde": np.asarray([[10.0, 2.0], [20.0, 5.0]], dtype=np.float32),
        "horizon": np.asarray([50, 50], dtype=np.int16),
        "hard": np.asarray([True, False]),
        "failure": np.asarray([False, False]),
    }
    selected, switch, selected_idx = pure._select({"max_switch": 0.0}, pred, ds)
    assert selected.tolist() == [10.0, 20.0]
    assert switch.tolist() == [False, False]
    assert selected_idx.tolist() == [0, 0]


def test_strict_gate_requires_positive_switch_and_easy_preservation():
    assert pure._strict_gate(
        {
            "all_improvement": 0.01,
            "t50_improvement": 0.01,
            "hard_failure_improvement": 0.01,
            "easy_degradation": 0.02,
            "switch_rate": 0.01,
        }
    )


def test_endpoint_residual_policy_prefers_conservative_tie_break(monkeypatch):
    policies = [
        {"type": "bounded_endpoint_residual", "mode": "all", "alpha": 0.15, "max_switch": 1.0},
        {
            "type": "bounded_endpoint_residual",
            "mode": "gain_harm",
            "alpha": 0.15,
            "gain_prob": 0.0,
            "harm_prob": 0.5,
            "physical_prob": 0.0,
            "max_switch": 1.0,
        },
    ]

    def fake_eval(_pred, _split, policy):
        # Validation utility is intentionally near-tied; the guarded policy has
        # slightly lower raw improvement but lower switch risk. This captures the
        # strict UCY repair: do not choose an ungated full-row residual when a
        # similarly good harm-gated policy is available.
        if policy["mode"] == "all":
            return {
                "all_improvement": 0.1000,
                "t50_improvement": 0.0900,
                "t100_improvement": 0.0800,
                "hard_failure_improvement": 0.1000,
                "easy_degradation": 0.0,
                "harm_over_fallback": -0.08,
                "switch_rate": 1.0,
            }
        return {
            "all_improvement": 0.0995,
            "t50_improvement": 0.0895,
            "t100_improvement": 0.0795,
            "hard_failure_improvement": 0.0995,
            "easy_degradation": 0.0,
            "harm_over_fallback": -0.08,
            "switch_rate": 0.90,
        }

    monkeypatch.setattr(pure, "_endpoint_residual_grid", lambda: policies)
    monkeypatch.setattr(pure, "_eval_endpoint_residual", fake_eval)

    policy, metrics = pure._select_endpoint_residual_policy({})
    assert policy["mode"] == "gain_harm"
    assert policy["selection_rule"] == "validation_score_minus_switch_ungated_easy_risk_penalty"
    assert metrics["switch_rate"] == 0.90
    assert not pure._strict_gate(
        {
            "all_improvement": 0.01,
            "t50_improvement": 0.01,
            "hard_failure_improvement": 0.01,
            "easy_degradation": 0.021,
            "switch_rate": 0.01,
        }
    )
