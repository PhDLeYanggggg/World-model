from pathlib import Path

from src import stage42_teacherless_gate_deployment_contract as hf
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/teacherless_gate_deployment_contract_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hf._build_payload()


def _decision(payload, request):
    return {row["request"]: row for row in payload["contract"]["requests"]}[request]


def test_stage42_hf_gate_passes() -> None:
    payload = _payload()
    gate = payload["stage42_hf_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hf_teacherless_gate_deployment_contract_pass"


def test_stage42_hf_allows_teacherless_but_keeps_floor() -> None:
    payload = _payload()
    allowed = _decision(payload, "teacherless_proximity_guarded_switch_gate")
    removal = _decision(payload, "teacher_gate_removal_for_repaired_gate")
    floor = _decision(payload, "causal_floor_removal")
    assert allowed["allowed"] is True
    assert allowed["status"] == "allowed_protected"
    assert removal["status"] == "allowed_policy_specific_not_global"
    assert floor["allowed"] is False
    assert payload["claim_boundary"]["causal_floor_safety_fallback_still_required"] is True
    assert payload["claim_boundary"]["global_floor_removal_allowed"] is False


def test_stage42_hf_blocks_overclaims_and_stage5c_smc() -> None:
    payload = _payload()
    assert _decision(payload, "ungated_neural_or_floor_free_global_deployment")["allowed"] is False
    assert _decision(payload, "metric_seconds_true3d_foundation_claim")["allowed"] is False
    assert _decision(payload, "stage5c_execution_or_smc_enabled")["allowed"] is False
    assert _decision(payload, "unknown_future_policy_request")["allowed"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
