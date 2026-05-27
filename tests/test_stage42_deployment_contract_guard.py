from __future__ import annotations

from src.stage42_deployment_contract_guard import _build_contract, _gate, evaluate_contract_request


def _dn() -> dict:
    return {
        "stage42_dn_gate": {"passed": 20, "total": 20},
        "deployment_variants": [
            {"variant": "proximity_guard", "deployment_status": "deployable_when_joint_proximity_safety_is_required"},
            {"variant": "no_proximity_guard", "deployment_status": "diagnostic_not_safety_sensitive"},
            {
                "variant": "group_consistency_full_waypoint_runtime",
                "deployment_status": "runtime_ready_for_its_source_level_protocol",
            },
        ],
    }


def _em() -> dict:
    return {
        "stage42_em_gate": {"passed": 14, "total": 14},
        "summary": {
            "targets": 5,
            "official_or_toolkit_source_candidates": 4,
            "manual_terms_required_targets": 5,
            "auto_download_allowed_now": 0,
            "conversion_ready_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
            "estimated_t50_after_terms": 10060,
            "estimated_t100_after_terms": 5696,
        },
    }


def _en() -> dict:
    return {
        "stage42_en_gate": {"passed": 13, "total": 13},
        "summary": {
            "floor_free_neural_deployable": False,
            "global_floor_removal_allowed": False,
            "teacher_floor_rollout_context_removal_allowed": False,
            "safe_partial_floor_relaxation_available": True,
            "partial_relaxation_components": ["t50_slice_relaxation::TrajNet|50", "t50_slice_relaxation::UCY|50"],
            "proximity_guard_required_for_safety_claim": True,
        },
        "component_decision_map": [
            {
                "component": "ungated_neural_endpoint_or_full_waypoint",
                "decision": "blocked",
                "key_metrics": {"ungated_endpoint_easy_degradation": 1.2459},
            },
            {
                "component": "teacher_floor_rollout_context",
                "decision": "required",
                "key_metrics": {"no_floor_rel_context_t50_delta": -0.0921},
            },
            {
                "component": "deployment_fallback_floor",
                "decision": "required_globally_partial_relaxation_allowed",
                "key_metrics": {"global_t50_improvement_after_repair": 0.2897},
            },
            {
                "component": "source_expansion_without_terms",
                "decision": "blocked",
                "key_metrics": {"conversion_ready_now": 0},
            },
        ],
    }


def _payload() -> dict:
    dn = _dn()
    em = _em()
    en = _en()
    return {
        "inputs": {
            "stage42_dn": {"stage42_dn_gate": dn["stage42_dn_gate"]},
            "stage42_em": {"stage42_em_gate": em["stage42_em_gate"]},
            "stage42_en": {"stage42_en_gate": en["stage42_en_gate"]},
            "stage42_eo": {"stage42_eo_gate": {"passed": 14, "total": 14}},
        },
        "contract": _build_contract(dn, em, en),
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_contract_allows_only_protected_or_diagnostic_requests() -> None:
    dn, em, en = _dn(), _em(), _en()

    assert evaluate_contract_request("safety_sensitive_bridge_shape_deployment", dn, em, en).allowed is True
    assert evaluate_contract_request("accuracy_priority_no_guard_reporting", dn, em, en).status == "allowed_diagnostic_only"
    assert evaluate_contract_request("source_level_full_waypoint_runtime", dn, em, en).status == "allowed_protocol_specific"
    assert evaluate_contract_request("validation_backed_t50_slice_relaxation", dn, em, en).status == "allowed_slice_only"


def test_contract_blocks_floor_free_source_conversion_and_metric_claims() -> None:
    dn, em, en = _dn(), _em(), _en()

    assert evaluate_contract_request("global_floor_free_neural_deployment", dn, em, en).allowed is False
    assert evaluate_contract_request("teacher_floor_rollout_context_removal", dn, em, en).allowed is False
    assert evaluate_contract_request("source_conversion_without_user_terms", dn, em, en).allowed is False
    assert evaluate_contract_request("metric_seconds_or_foundation_claim", dn, em, en).allowed is False
    assert evaluate_contract_request("new_unknown_policy", dn, em, en).status == "unknown_request_blocked_by_default"


def test_stage42_ep_gate_passes_for_complete_contract() -> None:
    gate = _gate(_payload())

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ep_deployment_contract_guard_pass"
