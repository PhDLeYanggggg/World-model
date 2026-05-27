from __future__ import annotations

import numpy as np

from src.stage42_group_consistency_target_ablation import _contribution_delta, _gate, _group_key_variants, _select_group_schema


def _row(schema: str, all_imp: float, t50: float, hard: float, easy: float, near: float, unsafe: int, val: float) -> dict:
    return {
        "group_schema": schema,
        "source": "fresh_run",
        "candidate_count": 72,
        "selected_candidate": {},
        "selected_val_score": val,
        "selected_val_metric": {},
        "metric_vs_floor": {
            "all_improvement": all_imp,
            "t50_improvement": t50,
            "t100_raw_frame_diagnostic_improvement": 0.1,
            "hard_failure_improvement": hard,
            "easy_degradation": easy,
        },
        "diagnostics": {
            "unsafe_rows": unsafe,
            "base_near_005": 0.02,
            "final_near_005": near,
            "final_p05_min_distance": 0.08,
        },
        "bootstrap": {"all": {"low": 0.1}, "hard_failure": {"low": 0.1}},
        "by_domain": {},
        "selection_score_on_test": val,
    }


def test_stage42_et_group_key_variants_include_controls() -> None:
    data = {
        "source_file": np.asarray(["a", "a", "b"], dtype=object),
        "frame_id": np.asarray([1.0, 1.0, 2.0]),
        "horizon": np.asarray([50, 50, 10]),
        "dataset": np.asarray(["UCY", "UCY", "TrajNet"], dtype=object),
        "agent_id": np.asarray([1, 2, 1]),
    }
    proper = np.asarray(["a\t1000\t50", "a\t1000\t50", "b\t2000\t10"], dtype=object)

    variants = _group_key_variants(data, proper)

    assert "source_frame_horizon" in variants
    assert "agent_isolated_no_interaction" in variants
    assert variants["agent_isolated_no_interaction"][0] != variants["agent_isolated_no_interaction"][1]
    assert len(variants["source_frame_horizon"]) == 3


def test_stage42_et_selection_keeps_source_frame_horizon_when_it_beats_isolated() -> None:
    correct = _row("source_frame_horizon", 0.25, 0.22, 0.24, -0.1, 0.01, 10, 1.0)
    isolated = _row("agent_isolated_no_interaction", 0.24, 0.20, 0.22, -0.1, 0.02, 0, 0.9)
    shuffled = _row("shuffled_source_frame_horizon", 0.23, 0.18, 0.20, -0.1, 0.03, 8, 0.8)

    selection = _select_group_schema([correct, isolated, shuffled])

    assert selection["selected_target_for_next_stage"] == "source_frame_horizon"
    assert selection["decision"] == "keep_source_frame_horizon_group_consistency_target"
    assert selection["source_frame_horizon_vs_agent_isolated"]["hard_failure_increment"] > 0
    assert selection["source_frame_horizon_vs_agent_isolated"]["near005_reduction_vs_correct_base"] > 0


def test_stage42_et_gate_passes_for_positive_correct_group() -> None:
    rows = [
        _row("source_frame_horizon", 0.25, 0.22, 0.24, -0.1, 0.01, 10, 1.0),
        _row("agent_isolated_no_interaction", 0.24, 0.20, 0.22, -0.1, 0.02, 0, 0.9),
        _row("source_frame_no_horizon", 0.23, 0.18, 0.20, -0.1, 0.02, 5, 0.7),
        _row("source_framebucket10_horizon", 0.22, 0.17, 0.19, -0.1, 0.02, 5, 0.6),
        _row("shuffled_source_frame_horizon", 0.21, 0.16, 0.18, -0.1, 0.03, 5, 0.5),
        _row("domain_frame_horizon", 0.20, 0.15, 0.17, -0.1, 0.03, 5, 0.4),
    ]
    payload = {
        "group_schema_ablation": {"rows": rows, "selection": _select_group_schema(rows)},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_candidate_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert _contribution_delta(rows[0], rows[1])["near005_reduction_vs_correct_base"] > 0
