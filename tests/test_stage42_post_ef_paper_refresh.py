from __future__ import annotations

from src.stage42_post_ef_paper_refresh import _gate, _paper_claim_matrix, _refresh_lines, _summary


def _ec() -> dict:
    return {
        "stage42_ec_gate": {"passed": 17, "total": 17},
        "summary": {
            "statistical_evidence": {
                "global_all_ci_low": 0.325616,
                "global_t50_ci_low": 0.265328,
                "global_hard_ci_low": 0.315115,
            }
        },
    }


def _ee() -> dict:
    return {
        "stage42_ee_gate": {"passed": 12, "total": 12},
        "summary": {
            "selected_candidate": "baseline_plus_knn_graph",
            "material_context_contribution": False,
            "selected_delta_all": 0.000368,
            "selected_delta_t50": -0.000074,
            "selected_delta_hard": 0.000424,
            "material_delta_threshold": 0.01,
        },
    }


def _ef() -> dict:
    return {
        "stage42_ef_gate": {"passed": 13, "total": 13},
        "summary": {
            "conversion_ready_now": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "estimated_t50_windows_after_terms": 10060,
            "estimated_t100_windows_after_terms": 5696,
            "top_unblock_targets": ["ucy_crowd_original", "eth_biwi_original"],
        },
    }


def _eb() -> dict:
    return {"stage42_eb_gate": {"passed": 12, "total": 12}}


def test_stage42_eg_claim_matrix_blocks_context_and_source_overclaims() -> None:
    rows = _paper_claim_matrix(_ec(), _ee(), _ef())
    by_claim = {row["claim"]: row for row in rows}

    assert by_claim["protected_source_level_group_consistency_full_waypoint"]["main_claim_allowed"] is True
    assert by_claim["current_context_switchability_scene_goal_neighbor_interaction"]["main_claim_allowed"] is False
    assert by_claim["source_conversion_metric_time_expansion"]["main_claim_allowed"] is False
    assert by_claim["global_metric_or_seconds_level_world_model"]["status"] == "forbidden"


def test_stage42_eg_refresh_lines_record_ee_ef_boundaries() -> None:
    lines = "\n".join(_refresh_lines(_summary(_eb(), _ec(), _ee(), _ef())))

    assert "Context main claim remains blocked" in lines
    assert "Source conversion remains blocked" in lines
    assert "protected source-level group-consistency full-waypoint dynamics" in lines
    assert "Still forbidden: true 3D" in lines


def test_stage42_eg_gate_passes_for_post_ee_ef_refresh() -> None:
    summary = _summary(_eb(), _ec(), _ee(), _ef())
    payload = {
        "inputs": {
            "stage42_eb": _eb(),
            "stage42_ec": _ec(),
            "stage42_ee": _ee(),
            "stage42_ef": _ef(),
        },
        "paper_refresh_summary": summary,
        "paper_file_status": [
            {
                "contains_stage42_eg": True,
                "contains_context_blocker": True,
                "contains_source_terms_blocker": True,
                "contains_group_claim": True,
                "contains_non_claims": True,
            }
        ],
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_eg_post_ee_ef_paper_refresh_pass"
