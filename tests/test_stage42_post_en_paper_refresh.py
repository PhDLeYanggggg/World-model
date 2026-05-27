from __future__ import annotations

from src.stage42_post_en_paper_refresh import _gate, _paper_claim_matrix, _refresh_lines, _summary


def _eg() -> dict:
    return {
        "stage42_eg_gate": {"passed": 12, "total": 12},
        "paper_refresh_summary": {
            "supported_main_claims": ["protected_source_level_group_consistency_full_waypoint"],
        },
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
            "next_validator_command": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
            "next_guarded_launcher_command": ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
        },
    }


def _en() -> dict:
    return {
        "stage42_en_gate": {"passed": 13, "total": 13},
        "summary": {
            "components_audited": 7,
            "floor_free_neural_deployable": False,
            "safe_partial_floor_relaxation_available": True,
            "global_floor_removal_allowed": False,
            "teacher_floor_rollout_context_removal_allowed": False,
            "proximity_guard_required_for_safety_claim": True,
            "partial_relaxation_components": ["t50_slice_relaxation::TrajNet|50"],
            "safety_required_components": ["teacher_floor_rollout_context", "deployment_fallback_floor", "proximity_guard"],
            "conversion_ready_now": 0,
        },
        "component_decision_map": [
            {
                "component": "ungated_neural_endpoint_or_full_waypoint",
                "decision": "blocked",
                "key_metrics": {"ungated_endpoint_easy_degradation": 1.2459, "easy_limit": 0.02},
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
                "component": "proximity_guard",
                "decision": "required_for_safety_sensitive_reporting",
                "key_metrics": {"guard_t50_improvement": 0.0107, "guard_near_collision_delta": -0.0006},
            },
            {
                "component": "source_expansion_without_terms",
                "decision": "blocked",
                "key_metrics": {"conversion_ready_now": 0, "auto_download_allowed_now": 0},
            },
        ],
    }


def test_stage42_eo_claim_matrix_preserves_source_and_floor_boundaries() -> None:
    rows = _paper_claim_matrix(_eg(), _em(), _en())
    by_claim = {row["claim"]: row for row in rows}

    assert by_claim["protected_source_level_group_consistency_full_waypoint"]["main_claim_allowed"] is True
    assert by_claim["official_source_expansion_or_conversion"]["main_claim_allowed"] is False
    assert by_claim["floor_free_neural_deployment"]["main_claim_allowed"] is False
    assert by_claim["teacher_floor_rollout_context_removal"]["main_claim_allowed"] is False
    assert by_claim["validation_backed_t50_slice_floor_relaxation"]["status"] == "partial_supported"
    assert by_claim["global_metric_seconds_foundation_or_stage5c_smc"]["status"] == "forbidden"


def test_stage42_eo_refresh_lines_record_manual_terms_and_floor_boundary() -> None:
    lines = "\n".join(_refresh_lines(_summary(_eg(), _em(), _en())))

    assert "Official links are not license acceptance" in lines
    assert "floor_free_neural_deployable" in lines
    assert "validation-backed t50 floor relaxation" in lines
    assert "Stage5C execution" in lines
    assert "SMC readiness" in lines


def test_stage42_eo_gate_passes_for_post_em_en_refresh() -> None:
    summary = _summary(_eg(), _em(), _en())
    payload = {
        "inputs": {
            "stage42_eg": {"stage42_eg_gate": _eg()["stage42_eg_gate"]},
            "stage42_em": {"stage42_em_gate": _em()["stage42_em_gate"]},
            "stage42_en": {"stage42_en_gate": _en()["stage42_en_gate"]},
        },
        "paper_refresh_summary": summary,
        "paper_file_status": [
            {
                "contains_stage42_eo": True,
                "contains_source_blocker": True,
                "contains_floor_blocker": True,
                "contains_partial_t50": True,
                "contains_proximity_guard": True,
                "contains_non_claims": True,
            }
        ],
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_eo_post_em_en_paper_refresh_pass"
