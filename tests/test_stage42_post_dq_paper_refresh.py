from __future__ import annotations

from pathlib import Path

from src import stage42_post_dq_paper_refresh as dr


def _summary() -> dict:
    return {
        "context_closure": {
            "decision": "close_current_sequence_graph_residual_context_protocol",
            "best_delta_all": -0.01,
            "best_delta_t50": -0.02,
            "best_delta_hard_failure": -0.03,
            "positive_context_rows": [],
        },
        "full_waypoint_checkpoint": {
            "runtime_all_improvement": 0.2,
            "runtime_t50_improvement": 0.1,
            "runtime_t100_raw_frame_diagnostic_improvement": 0.05,
            "runtime_hard_failure_improvement": 0.2,
            "runtime_easy_degradation": 0.0,
            "runtime_switch_rate": 0.5,
            "switch_exact_match": True,
            "selected_xy_max_abs_diff": 0.0,
            "runtime_base_near_005": 0.02,
            "runtime_final_near_005": 0.01,
            "runtime_floor_near_005": 0.03,
        },
        "deployment_variant_boundary": {
            "safety_sensitive_default": "proximity_guard",
            "accuracy_priority_diagnostic": "no_proximity_guard",
            "source_level_full_waypoint_runtime_candidate": "group_consistency_full_waypoint_runtime",
            "baseline_mixing_caveat": True,
        },
        "source_legal_time_boundary": {
            "conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "global_metric_seconds_claim_allowed": False,
            "global_t100_deployable_claim_allowed": False,
        },
        "paper_verdict": {
            "paper_ready_evidence_package_strengthened": True,
            "context_main_claim_allowed": False,
            "full_waypoint_runtime_evidence_allowed": True,
            "ungated_neural_or_full_waypoint_deployment_allowed": False,
            "metric_seconds_claim_allowed": False,
            "foundation_claim_allowed": False,
            "stage5c_execution_allowed": False,
            "smc_allowed": False,
        },
    }


def test_refresh_lines_include_context_full_waypoint_and_boundaries() -> None:
    text = "\n".join(dr._refresh_lines(_summary()))
    assert "sequence/graph/neighbor/goal context remains auxiliary or diagnostic" in text
    assert "group-consistency full-waypoint runtime policy is valid evidence" in text
    assert "dataset-local/raw-frame only" in text
    assert "Do not claim Stage5C execution" in text


def test_refresh_paper_files_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Paper\n", encoding="utf-8")
    monkeypatch.setattr(dr, "PAPER_FILES", [paper])
    first = dr._refresh_paper_files(_summary())
    second = dr._refresh_paper_files(_summary())
    text = paper.read_text(encoding="utf-8")
    assert first[0]["contains_stage42_dr"]
    assert second[0]["contains_stage42_dr"]
    assert text.count("STAGE42_DR_POST_DQ_PAPER_REFRESH:START") == 1


def test_gate_passes_for_honest_post_dq_refresh() -> None:
    payload = {
        "inputs": {
            "stage42_dp": {"stage42_dp_gate": {"passed": 19, "total": 19}},
            "stage42_dq": {"stage42_dq_gate": {"passed": 24, "total": 24}},
            "stage42_dn": {"stage42_dn_gate": {"passed": 20, "total": 20}},
            "stage42_do": {"stage42_do_gate": {"passed": 13, "total": 13}},
        },
        "paper_refresh_summary": _summary(),
        "paper_file_status": [{"contains_stage42_dr": True}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = dr._gate(payload)
    assert gate["verdict"] == "stage42_dr_post_dq_paper_refresh_pass"
    assert gate["passed"] == gate["total"]
