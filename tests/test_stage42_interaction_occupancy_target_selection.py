from __future__ import annotations

from src.stage42_interaction_occupancy_target_selection import _gate, _select_target, _selection_score


def _row(name: str, all_imp: float, t50: float, hard: float, easy: float, delta_all: float, delta_hard: float, near_gain: bool = False) -> dict:
    near = None
    if near_gain:
        near = {"base_near_005": 0.02, "final_near_005": 0.01}
    return {
        "target_family": name,
        "source": "fresh_run",
        "source_gate_verdict": "ok",
        "source_gate_passed": 17,
        "source_gate_total": 17,
        "selected": {},
        "metric_vs_floor": {
            "all_improvement": all_imp,
            "t50_improvement": t50,
            "t100_raw_frame_diagnostic_improvement": 0.01,
            "hard_failure_improvement": hard,
            "easy_degradation": easy,
        },
        "delta_vs_stage42_am": {
            "all_improvement": delta_all,
            "hard_failure_improvement": delta_hard,
        },
        "deployment_decision": {"promote_group_consistency_full_waypoint_repair": near_gain},
        "near_diagnostics": near,
        "promotable": near_gain,
    }


def test_stage42_es_selection_prefers_promotable_group_consistency() -> None:
    scalar = _row("scalar_proximity_occupancy_loss", 0.25, 0.22, 0.24, -0.2, 0.0, -0.001)
    group = _row("explicit_group_consistency_repair", 0.247, 0.224, 0.239, -0.25, 0.001, 0.001, near_gain=True)

    selected = _select_target([scalar, group])

    assert selected["selected_target_family"] == "explicit_group_consistency_repair"
    assert selected["decision"] == "continue_with_explicit_group_consistency_interaction_target"
    assert _selection_score(group) > _selection_score(scalar)


def test_stage42_es_gate_passes_for_group_selected_no_leakage() -> None:
    scalar = _row("scalar_proximity_occupancy_loss", 0.25, 0.22, 0.24, -0.2, 0.0, -0.001)
    group = _row("explicit_group_consistency_repair", 0.247, 0.224, 0.239, -0.25, 0.001, 0.001, near_gain=True)
    selected = _select_target([scalar, group])
    payload = {
        "fresh_inputs": {
            "stage42_dh_rerun": {"gate": {"passed": 15, "total": 16}},
            "stage42_di_rerun": {"gate": {"passed": 17, "total": 17}},
        },
        "target_selection": selected,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_model_or_policy_selection": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_es_interaction_occupancy_target_selection_pass"


def test_stage42_es_score_penalizes_easy_harm() -> None:
    safe = _row("explicit_group_consistency_repair", 0.1, 0.1, 0.1, 0.0, 0.02, 0.02, near_gain=True)
    unsafe = _row("explicit_group_consistency_repair", 0.1, 0.1, 0.1, 0.08, 0.02, 0.02, near_gain=True)

    assert _selection_score(safe) > _selection_score(unsafe)
