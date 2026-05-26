import numpy as np

from src import stage42_explicit_gain_harm_selector as s42o


def test_stage42o_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42o.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42o_selector_switch_uses_scores_not_easy_labels() -> None:
    scores = {
        "switch_prob": np.asarray([0.9, 0.9, 0.2, 0.9], dtype=np.float32),
        "gain": np.asarray([0.5, 0.5, 0.5, 0.01], dtype=np.float32),
        "harm_prob": np.asarray([0.1, 0.8, 0.1, 0.1], dtype=np.float32),
        "uncertainty": np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
    }
    labels = {
        "domain": np.asarray(["A", "A", "A", "A"]),
        "horizon": np.asarray([50, 50, 50, 50]),
        # If the policy accidentally uses this, the first row would be blocked.
        "easy": np.asarray([True, False, False, False]),
    }
    policy = {
        "slices": {
            "A|50": {
                "switch_min": 0.5,
                "gain_min": 0.05,
                "harm_max": 0.3,
                "uncertainty_max": 0.5,
                "max_switch": 1.0,
            }
        }
    }
    assert s42o._selector_switch(scores, labels, policy).tolist() == [True, False, False, False]


def test_stage42o_gate_requires_policy_without_easy_label_input() -> None:
    base = {
        "rows": [
            {"seed": 131, "base_info": {"source": "cached_verified_stage42n"}},
            {"seed": 137, "base_info": {"source": "cached_verified_stage42n"}},
            {"seed": 139, "base_info": {"source": "cached_verified_stage42n"}},
        ],
        "summary": {
            "seeds": [131, 137, 139],
            "ade_all": {"mean": 0.03},
            "ade_t50": {"mean": 0.01},
            "ade_hard_failure": {"mean": 0.03},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "comparison": {"stage42_n_row_gain_static_gate": {"ade_t50": {"mean": -0.02}}},
        "source_labels": {"row_teacher_test": "not_built", "policy_uses_easy_label": False},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    assert s42o._gate(base)["passed"] == s42o._gate(base)["total"]
    base["source_labels"]["policy_uses_easy_label"] = True
    assert s42o._gate(base)["gates"]["policy_no_easy_label_input"] is False
    base["source_labels"]["policy_uses_easy_label"] = False
    base["no_leakage"]["test_statistics_normalization"] = True
    assert s42o._gate(base)["gates"]["no_test_statistics_normalization"] is False
