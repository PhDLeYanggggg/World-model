from src import stage42_fresh_static_gated_checkpoint as s42k


def test_stage42k_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42k.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42k_stat_ci_has_expected_keys() -> None:
    stat = s42k._stat([0.1, 0.2, 0.3])
    assert set(stat) == {"mean", "std", "ci_low", "ci_high"}
    assert stat["ci_low"] <= stat["mean"] <= stat["ci_high"]


def test_stage42k_gate_accepts_fresh_static_checkpoint() -> None:
    result = {
        "rows": [{"train_info": {"source": "fresh_run"}}, {"train_info": {"source": "fresh_run"}}, {"train_info": {"source": "fresh_run"}}],
        "summary": {
            "seeds": [71, 73, 79],
            "ade_all": {"mean": 0.02},
            "ade_t50": {"mean": 0.03},
            "ade_hard_failure": {"mean": 0.02},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "comparison": {
            "stage42_j_full_static": {
                "ade_all": {"mean": -0.01},
                "ade_t50": {"mean": -0.02},
            }
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42k._gate(result)
    assert gate["passed"] == gate["total"]
