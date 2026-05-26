import pytest

from src import stage42_full_waypoint_auxiliary_ablation as s42ab


def test_stage42ab_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42ab.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42ab_delta_vs_full_direction() -> None:
    stage42i = {
        "summary": {
            "sequence_waypoint_full": {
                "ade_all": {"mean": 0.20},
                "ade_t50": {"mean": 0.15},
                "ade_hard_failure": {"mean": 0.18},
                "fde_t50": {"mean": 0.30},
            }
        }
    }
    no_aux = {
        "ade_all": {"mean": 0.10},
        "ade_t50": {"mean": 0.05},
        "ade_hard_failure": {"mean": 0.12},
        "fde_t50": {"mean": 0.25},
    }
    delta = s42ab._delta_vs_full(no_aux, stage42i)
    assert delta["ade_all_delta_full_minus_no_aux"] == pytest.approx(0.10)
    assert delta["ade_t50_delta_full_minus_no_aux"] == pytest.approx(0.10)
    assert delta["ade_hard_delta_full_minus_no_aux"] == pytest.approx(0.06)
    assert delta["fde_t50_delta_full_minus_no_aux"] == pytest.approx(0.05)


def test_stage42ab_gate_accepts_mixed_evidence_when_not_overclaimed() -> None:
    result = {
        "source": "fresh_run",
        "inputs": {"stage42i_exists": True},
        "summary": {
            "seeds": [67, 71, 73],
            "ade_easy_degradation": {"mean": 0.0},
        },
        "delta_vs_stage42i_full": {
            "ade_all_delta_full_minus_no_aux": -0.01,
            "ade_t50_delta_full_minus_no_aux": 0.02,
            "ade_hard_delta_full_minus_no_aux": -0.03,
            "fde_t50_delta_full_minus_no_aux": 0.01,
        },
        "ablation_protocol": {
            "same_model_outputs": True,
            "interaction_occupancy_physical_loss_weight": 0.0,
        },
        "interpretation": {"uniform_aux_positive_claim_allowed": False},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42ab._gate(result)
    assert gate["passed"] == gate["total"]
