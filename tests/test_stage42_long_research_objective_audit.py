from pathlib import Path

from src import stage42_long_research_objective_audit as ho
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/long_research_objective_audit_stage42.json")
    if path.exists():
        return read_json(path, {})
    return ho.build_payload()


def test_stage42_ho_keeps_long_goal_active() -> None:
    payload = _payload()
    overall = payload["coverage"]["overall_stage42_long_goal"]
    assert overall["status"] == "active_not_complete"
    assert overall["pass_for_objective"] is False


def test_stage42_ho_blocks_metric_time_until_ready_candidates() -> None:
    payload = _payload()
    metric_time = payload["metric_time"]
    stage_a = payload["coverage"]["stage_a_data_and_calibration"]
    assert metric_time["ready_candidates"] == 0
    assert metric_time["conversion_queue_count"] == 0
    assert metric_time["restricted_metric_time_claim_allowed_now"] is False
    assert stage_a["status"] == "partial_blocked"


def test_stage42_ho_blocks_context_overclaim() -> None:
    payload = _payload()
    context = payload["context"]
    stage_d = payload["coverage"]["stage_d_causal_ablation"]
    assert context["scene_goal_independent_claim"] == "not_supported"
    assert context["neighbor_interaction_independent_claim"] == "not_supported"
    assert stage_d["pass_for_objective"] is False
    assert "residual context variants underperformed" in context["reason"]


def test_stage42_ho_records_supported_full_waypoint_path() -> None:
    payload = _payload()
    stage_c = payload["coverage"]["stage_c_full_waypoint_dynamics"]
    assert stage_c["status"] == "protected_positive_not_ungated"
    assert stage_c["pass_for_objective"] is True
    assert payload["full_waypoint"]["group_consistency_supported"] is True


def test_stage42_ho_gate_passes_as_audit_not_completion() -> None:
    payload = _payload()
    gate = payload["stage42_ho_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ho_long_research_objective_audit_pass_keep_goal_active"
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
