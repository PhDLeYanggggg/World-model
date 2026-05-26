from src import stage42_policy_distilled_static_gate as s42m


def test_stage42m_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42m.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42m_alpha_for_expert() -> None:
    assert s42m._alpha_for_expert("floor") == 0.0
    assert s42m._alpha_for_expert("static_alpha025") == 0.25
    assert s42m._alpha_for_expert("static_alpha050") == 0.50
    assert s42m._alpha_for_expert("full_static") == 1.0
    assert s42m._alpha_for_expert("unknown") == 0.0


def test_stage42m_gate_requires_teacher_source_and_t50_gain() -> None:
    base = {
        "rows": [{"seed": 101}, {"seed": 103}, {"seed": 107}],
        "summary": {
            "seeds": [101, 103, 107],
            "ade_all": {"mean": 0.03},
            "ade_t50": {"mean": 0.01},
            "ade_hard_failure": {"mean": 0.03},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "comparison": {"stage42_l_horizon_static_gate": {"ade_t50": {"mean": 0.002}}},
        "source_labels": {"teacher_policy": "cached_verified_stage42j_validation_selected_no_test_endpoints"},
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
    assert s42m._gate(base)["passed"] == s42m._gate(base)["total"]
    base["source_labels"]["teacher_policy"] = "bad_teacher"
    assert s42m._gate(base)["gates"]["teacher_source_no_test"] is False
