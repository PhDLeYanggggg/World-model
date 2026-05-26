from __future__ import annotations

from pathlib import Path

from src import stage42_full_waypoint_bridge_shape_audit as cm


def _metric(all_value: float, t50: float, t100: float, hard: float, easy: float = 0.0) -> dict:
    return {
        "rows": 10,
        "all_improvement": all_value,
        "t50_improvement": t50,
        "t100_improvement": t100,
        "t100_raw_frame_diagnostic_improvement": t100,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
        "switch_rate": 0.2,
    }


def test_replace_section_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("intro\n", encoding="utf-8")
    cm._replace_section(path, "X", ["old"])
    cm._replace_section(path, "X", ["new"])
    text = path.read_text(encoding="utf-8")
    assert "old" not in text
    assert "new" in text
    assert text.count("X:START") == 1


def test_deltas_capture_horizon_lift_and_all_tradeoff() -> None:
    full = {
        "comparisons": {
            "m3w_neural_v1_composite_tail_linear_bridge": {"ade": _metric(0.21, 0.13, 0.14, 0.20)},
            "full_waypoint_transformer_protected": {"ade": _metric(0.18, 0.15, 0.22, 0.19)},
            "graph_interaction_group_consistency": {
                "report": {"test_metrics": _metric(0.22, 0.16, 0.23, 0.22)}
            },
        }
    }
    full["comparisons"]["graph_interaction_group_consistency"]["report"]["test_metrics"][
        "collision_delta_vs_floor_005"
    ] = 0.01
    deltas = cm._deltas(full)
    assert deltas["full_waypoint_minus_linear_bridge"]["all_improvement"] < 0.0
    assert deltas["full_waypoint_minus_linear_bridge"]["t50_improvement"] > 0.0
    assert deltas["full_waypoint_minus_linear_bridge"]["t100_raw_frame_diagnostic_improvement"] > 0.0


def test_gate_passes_with_safe_full_waypoint_boundary() -> None:
    full = {
        "stage42_c_gate": {
            "passed": 12,
            "total": 12,
            "gates": {"two_external_domains_positive": True},
        },
        "comparisons": {
            "endpoint_only_final_fde": {"claim_boundary": {"full_waypoint_model": False}},
            "m3w_neural_v1_composite_tail_linear_bridge": {"ade": _metric(0.21, 0.13, 0.14, 0.20)},
            "full_waypoint_transformer_protected": {"ade": _metric(0.18, 0.15, 0.22, 0.19)},
            "ungated_full_waypoint_transformer": {"ade": _metric(0.30, 0.25, 0.30, 0.30, easy=1.2)},
            "graph_interaction_group_consistency": {
                "report": {"test_metrics": {**_metric(0.22, 0.16, 0.23, 0.22), "collision_delta_vs_floor_005": 0.01}}
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "test_threshold_tuning": False,
        },
    }
    payload = {
        "source": "unit",
        "inputs": {
            "full_waypoint": full,
            "unified_row_cache": {"stage42_x_gate": {"passed": 16, "total": 16}},
            "ucy_bridge": {"verdict": "stage42_u_ucy_endpoint_to_full_bridge_failed_blocker"},
        },
        "paper_file_status": [{"contains_stage42_cm": True}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cm._gate(payload)
    assert gate["verdict"] == "stage42_cm_full_waypoint_bridge_shape_audit_pass"
    assert gate["passed"] == gate["total"]
