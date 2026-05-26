from __future__ import annotations

from pathlib import Path

from src import stage42_post_context_guard_paper_refresh as cl


def _variant(all_value: float, t50: float, hard: float) -> dict:
    return {
        "protected": {
            "all_improvement": all_value,
            "t50_improvement": t50,
            "t100_raw_frame_diagnostic_improvement": 0.1,
            "hard_failure_improvement": hard,
            "easy_degradation": -0.01,
            "switch_rate": 0.5,
        }
    }


def test_replace_section_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("intro\n", encoding="utf-8")
    cl._replace_section(path, "X", ["first"])
    cl._replace_section(path, "X", ["second"])
    text = path.read_text(encoding="utf-8")
    assert "first" not in text
    assert "second" in text
    assert text.count("X:START") == 1


def test_evidence_rows_block_context_overclaims() -> None:
    cj = {
        "stage42_cj_gate": {"passed": 10, "total": 10},
        "validation_only_selection": {"selected_variant": "baseline_family_control"},
        "goal_scene_rescue_success": False,
        "variants": {
            "baseline_family_control": _variant(0.28, 0.31, 0.27),
            "baseline_plus_goal_scene": _variant(0.26, 0.22, 0.24),
            "baseline_plus_motion_goal_context": _variant(0.24, 0.21, 0.23),
        },
    }
    ck = {
        "stage42_ck_gate": {"passed": 11, "total": 11},
        "validation_only_selection": {"selected_variant": "baseline_family_control"},
        "neighbor_interaction_rescue_success": False,
        "graph_info": {"graph_stats": {"rows": 100, "rows_with_neighbors": 95}},
        "variants": {
            "baseline_family_control": _variant(0.28, 0.31, 0.27),
            "baseline_plus_scalar_neighbor": _variant(0.26, 0.22, 0.24),
            "baseline_plus_knn_graph": _variant(0.24, 0.21, 0.23),
            "baseline_plus_graph_goal": _variant(0.20, 0.20, 0.18),
        },
    }
    rows = cl._evidence_rows(cj, ck)
    assert any(row["item"] == "Stage42-CJ goal/scene gated expert" for row in rows)
    assert any(row["item"] == "Stage42-CK neighbor/interaction gated expert" for row in rows)
    assert all(row["status"] == "diagnostic_negative" for row in rows)
    assert "goal_scene_rescue_success=False" in rows[0]["evidence"]
    assert "neighbor_interaction_rescue_success=False" in rows[2]["evidence"]


def test_gate_passes_when_cj_ck_are_not_overclaimed() -> None:
    payload = {
        "source": "unit",
        "inputs_loaded": {
            "cj": {
                "stage42_cj_gate": {"passed": 10, "total": 10},
                "goal_scene_rescue_success": False,
                "validation_only_selection": {"selected_variant": "baseline_family_control"},
                "no_leakage": {"test_threshold_tuning": False, "future_endpoint_input": False},
            },
            "ck": {
                "stage42_ck_gate": {"passed": 11, "total": 11},
                "neighbor_interaction_rescue_success": False,
                "validation_only_selection": {"selected_variant": "baseline_family_control"},
                "no_leakage": {"test_threshold_tuning": False, "future_endpoint_input": False},
            },
        },
        "paper_file_status": [{"contains_stage42_cl": True}],
        "claim_boundary": {
            "goal_scene_main_claim_allowed": False,
            "neighbor_interaction_main_claim_allowed": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cl._gate(payload)
    assert gate["verdict"] == "stage42_cl_context_guard_paper_refresh_pass"
    assert gate["passed"] == gate["total"]
