from __future__ import annotations

import json
from pathlib import Path

from src import stage42_full_waypoint_deployment_gap_audit as de


def _payloads() -> dict:
    return {
        "full_waypoint_boundary": {
            "source": "fresh_synthesis_from_stage42_full_waypoint_artifacts",
            "stage": "Stage42-CM endpoint bridge / full-waypoint shape audit",
            "comparison_rows": [
                {
                    "name": "m3w_neural_v1_composite_tail_linear_bridge",
                    "all_improvement": 0.21,
                    "t50_improvement": 0.13,
                    "t100_raw_frame_diagnostic_improvement": 0.14,
                    "hard_failure_improvement": 0.20,
                    "easy_degradation": -0.10,
                },
                {
                    "name": "full_waypoint_transformer_protected",
                    "all_improvement": 0.18,
                    "t50_improvement": 0.15,
                    "t100_raw_frame_diagnostic_improvement": 0.22,
                    "hard_failure_improvement": 0.19,
                    "easy_degradation": 0.0,
                },
                {
                    "name": "ungated_full_waypoint_transformer",
                    "all_improvement": 0.29,
                    "t50_improvement": 0.21,
                    "t100_raw_frame_diagnostic_improvement": 0.35,
                    "hard_failure_improvement": 0.32,
                    "easy_degradation": 1.2,
                },
                {
                    "name": "graph_interaction_group_consistency",
                    "all_improvement": 0.22,
                    "t50_improvement": 0.15,
                    "t100_raw_frame_diagnostic_improvement": 0.23,
                    "hard_failure_improvement": 0.22,
                    "easy_degradation": 0.0,
                },
                {
                    "name": "unified_row_level_full_waypoint_cache",
                    "all_improvement": 0.09,
                    "t50_improvement": 0.06,
                    "t100_raw_frame_diagnostic_improvement": 0.08,
                    "hard_failure_improvement": 0.09,
                    "easy_degradation": 0.001,
                },
            ],
            "deltas": {
                "full_waypoint_minus_linear_bridge": {
                    "all_improvement": -0.024,
                    "t50_improvement": 0.011,
                    "t100_raw_frame_diagnostic_improvement": 0.08,
                    "hard_failure_improvement": -0.008,
                },
                "graph_group_minus_full_waypoint": {
                    "all_improvement": 0.03,
                    "t50_improvement": 0.002,
                    "t100_raw_frame_diagnostic_improvement": 0.001,
                    "hard_failure_improvement": 0.02,
                    "collision_delta_vs_floor_005": 0.008,
                },
            },
            "stage42_cm_gate": {"verdict": "pass"},
        },
        "common_validation_composer": {
            "source": "fresh_common_validation_eval_from_cached_verified_checkpoints",
            "test_eval": {
                "metric_vs_endpoint_ade": {
                    "all_improvement": 0.03,
                    "t50_improvement": 0.015,
                    "t100_raw_frame_diagnostic_improvement": 0.06,
                    "hard_failure_improvement": 0.03,
                    "easy_degradation": 0.002,
                }
            },
            "stage42_co_gate": {"verdict": "pass"},
        },
        "proximity_guard": {
            "source": "fresh_validation_selected_proximity_guard_from_stage42_co_policy",
            "test_eval": {
                "metric_vs_endpoint_ade": {
                    "all_improvement": 0.017,
                    "t50_improvement": 0.010,
                    "t100_raw_frame_diagnostic_improvement": 0.034,
                    "hard_failure_improvement": 0.019,
                    "easy_degradation": 0.002,
                }
            },
            "test_joint_safety": {"composer_minus_endpoint": {"near_collision_rate_005": -0.0006}},
            "stage42_cq_gate": {"verdict": "pass"},
        },
        "proximity_ablation": {
            "source": "fresh_synthesis_from_stage42_co_cp_cq_artifacts",
            "ablation_rows": {
                "no_proximity_guard": {
                    "all_improvement": 0.03,
                    "t50_improvement": 0.015,
                    "t100_raw_frame_diagnostic_improvement": 0.06,
                    "hard_failure_improvement": 0.03,
                    "easy_degradation": 0.002,
                    "near_collision_005_delta_vs_endpoint": 0.003,
                },
                "proximity_guard": {
                    "all_improvement": 0.017,
                    "t50_improvement": 0.010,
                    "t100_raw_frame_diagnostic_improvement": 0.034,
                    "hard_failure_improvement": 0.019,
                    "easy_degradation": 0.002,
                    "near_collision_005_delta_vs_endpoint": -0.0006,
                },
            },
            "stage42_cr_gate": {"verdict": "pass"},
        },
        "unified_row_cache": {
            "source": "fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions",
            "summary": {
                "ade_all": {"mean": 0.09},
                "ade_t50": {"mean": 0.06},
                "ade_t100_raw_frame_diagnostic": {"mean": 0.08},
                "ade_hard_failure": {"mean": 0.09},
                "ade_easy_degradation": {"mean": 0.001},
            },
            "stage42_x_gate": {"verdict": "pass"},
        },
        "source_level_full_waypoint": {
            "source": "fresh_run",
            "model": {
                "metrics": {
                    "protected_ridge_source_level": {"all_improvement": 0.24},
                    "protected_ridge_source_level_fde": {"t50_improvement": 0.20},
                }
            },
            "stage42_am_gate": {"verdict": "pass"},
        },
        "source_support_closure": {
            "source": "fresh_stage42_dd_source_support_closure_audit",
            "closure_summary": {"domains_not_closed": ["ETH_UCY", "TrajNet", "UCY"]},
            "stage42_dd_gate": {"verdict": "pass"},
        },
    }


def test_evidence_summary_blocks_primary_promotion_but_keeps_guarded_composer() -> None:
    evidence = de._summarize_evidence(_payloads())
    support = evidence["support_flags"]
    decision = evidence["deployment_decision"]
    assert support["horizon_auxiliary_supported"] is True
    assert support["endpoint_linear_replacement_supported"] is False
    assert support["guarded_composer_supported"] is True
    assert support["ungated_full_waypoint_blocked"] is True
    assert decision["promote_full_waypoint_as_primary_deployable_dynamics"] is False
    assert "protected_full_waypoint_does_not_beat_endpoint_linear_on_all_and_hard" in decision["blockers"]


def test_gate_passes_for_honest_negative_deployment_decision() -> None:
    result = {
        "input_status": {
            key: {"exists": True, "source": value["source"], "stage": "", "generated_at_utc": "", "gate_verdict": ""}
            for key, value in _payloads().items()
        },
        "evidence": de._summarize_evidence(_payloads()),
        "no_leakage_and_protocol": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = de._gate(result)
    assert gate["verdict"] == "stage42_de_full_waypoint_deployment_gap_audit_pass_primary_promotion_blocked"
    assert gate["passed"] == gate["total"]


def test_run_writes_isolated_outputs(tmp_path: Path, monkeypatch) -> None:
    paths = {key: tmp_path / f"{key}.json" for key in de.INPUTS}
    for key, path in paths.items():
        path.write_text(json.dumps(_payloads()[key]), encoding="utf-8")
    monkeypatch.setattr(de, "INPUTS", paths)
    monkeypatch.setattr(de, "REPORT_JSON", tmp_path / "full_waypoint_deployment_gap_audit_stage42.json")
    monkeypatch.setattr(de, "REPORT_MD", tmp_path / "full_waypoint_deployment_gap_audit_stage42.md")
    monkeypatch.setattr(de, "GATE_MD", tmp_path / "stage42_stage_de_gate.md")
    result = de.run_stage42_full_waypoint_deployment_gap_audit(refresh_readmes=False)
    assert result["stage42_de_gate"]["verdict"] == "stage42_de_full_waypoint_deployment_gap_audit_pass_primary_promotion_blocked"
    assert de.REPORT_JSON.exists()
    assert de.REPORT_MD.exists()
    assert de.GATE_MD.exists()
