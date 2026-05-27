from __future__ import annotations

from src.stage42_floor_removability_decision_map import _component_map, _gate, _summary


def _inputs() -> dict:
    return {
        "bw": {
            "summary": {"floor_free_neural_deployable": False},
            "safety_floor_findings": {
                "ungated_endpoint": {"easy_degradation": 1.0},
                "ungated_full_waypoint": {"easy_degradation": 1.0},
            },
            "context_findings": {
                "no_floor_rel_context_protected_delta_t50": -0.1,
                "no_safe_baseline_context_protected_delta_t50": -0.1,
            },
        },
        "by": {
            "summary": {
                "repaired_t50_slices": ["TrajNet|50", "UCY|50"],
                "global_t50_improvement": 0.2,
                "global_easy_degradation": -0.1,
            },
            "target_t50_decisions": {
                "TrajNet|50": {
                    "protected_t50_repaired": True,
                    "before_bx_reason": "validation-backed",
                    "after_metric": {"rows": 10, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0, "switch_rate": 0.5},
                },
                "UCY|50": {
                    "protected_t50_repaired": True,
                    "before_bx_reason": "internal validation",
                    "after_metric": {"rows": 10, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0, "switch_rate": 0.5},
                },
            },
        },
        "cr": {
            "ablation_rows": {
                "no_proximity_guard": {"all_improvement": 0.03, "near_collision_005_delta_vs_endpoint": 0.003},
                "proximity_guard": {"all_improvement": 0.02, "t50_improvement": 0.01, "near_collision_005_delta_vs_endpoint": -0.001},
            }
        },
        "cq": {},
        "em": {"summary": {"official_or_toolkit_source_candidates": 4, "conversion_ready_now": 0, "auto_download_allowed_now": 0}},
    }


def test_stage42_en_component_map_blocks_global_floor_removal() -> None:
    components = _component_map(_inputs())
    decisions = {row["component"]: row["decision"] for row in components}

    assert decisions["ungated_neural_endpoint_or_full_waypoint"] == "blocked"
    assert decisions["teacher_floor_rollout_context"] == "required"
    assert decisions["t50_slice_relaxation::TrajNet|50"] == "partial_supported"


def test_stage42_en_summary_marks_partial_relaxation_only() -> None:
    summary = _summary(_component_map(_inputs()))

    assert summary["floor_free_neural_deployable"] is False
    assert summary["global_floor_removal_allowed"] is False
    assert summary["safe_partial_floor_relaxation_available"] is True
    assert summary["teacher_floor_rollout_context_removal_allowed"] is False


def test_stage42_en_gate_passes_with_floor_boundaries() -> None:
    components = _component_map(_inputs())
    payload = {
        "input_reports": {key: {"exists": True} for key in ["bw", "by", "cq", "cr", "em"]},
        "component_decision_map": components,
        "summary": _summary(components),
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
