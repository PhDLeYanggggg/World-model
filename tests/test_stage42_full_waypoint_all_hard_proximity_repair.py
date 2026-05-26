from __future__ import annotations

from pathlib import Path

import numpy as np

from src import stage42_full_waypoint_all_hard_proximity_repair as df


def _bundle() -> dict:
    labels = {
        "domain": np.array(["ETH_UCY"] * 4 + ["TrajNet"] * 4),
        "horizon": np.array([50, 50, 100, 100, 50, 50, 100, 100]),
        "hard": np.array([True, False, True, False, True, False, True, False]),
        "failure": np.array([False] * 8),
        "easy": np.array([False, True, False, True, False, True, False, True]),
        "normalizer": np.ones(8, dtype=np.float64),
        "current_xy": np.zeros((8, 2), dtype=np.float64),
        "waypoint_xy": np.ones((8, 4, 2), dtype=np.float64) * 0.75,
        "waypoint_valid": np.ones((8, 4), dtype=bool),
    }
    endpoint = {
        "labels": labels,
        "selected_ade": np.ones(8, dtype=np.float64),
        "selected_fde": np.ones(8, dtype=np.float64),
        "floor_ade": np.ones(8, dtype=np.float64) * 1.2,
        "floor_fde": np.ones(8, dtype=np.float64) * 1.2,
        "selected_xy": np.zeros((8, 4, 2), dtype=np.float64),
        "floor_xy": np.zeros((8, 4, 2), dtype=np.float64),
        "switch": np.zeros(8, dtype=bool),
    }
    full = {
        "labels": labels,
        "selected_ade": np.array([0.8, 1.0, 0.85, 1.0, 0.9, 1.0, 0.92, 1.0], dtype=np.float64),
        "selected_fde": np.array([0.7, 1.0, 0.8, 1.0, 0.85, 1.0, 0.9, 1.0], dtype=np.float64),
        "selected_xy": np.ones((8, 4, 2), dtype=np.float64),
    }
    return {"endpoint": endpoint, "full": full, "keys": np.arange(8), "alignment": {"aligned": True}}


def test_choices_require_all_hard_and_easy_guard() -> None:
    stats = {
        "ETH_UCY|50": {
            "rows": 100,
            "all_gain_vs_endpoint": 0.03,
            "hard_gain_vs_endpoint": 0.05,
            "treat_easy_degradation_vs_endpoint": 0.0,
        },
        "TrajNet|50": {
            "rows": 100,
            "all_gain_vs_endpoint": 0.03,
            "hard_gain_vs_endpoint": -0.01,
            "treat_easy_degradation_vs_endpoint": 0.0,
        },
        "UCY|50": {
            "rows": 10,
            "all_gain_vs_endpoint": 0.2,
            "hard_gain_vs_endpoint": 0.2,
            "treat_easy_degradation_vs_endpoint": 0.0,
        },
    }
    choices = df._choices_from_stats(stats, min_all_gain=0.0, min_hard_gain=0.0, easy_max=0.02)
    assert choices["ETH_UCY|50"] is True
    assert choices["TrajNet|50"] is False
    assert choices["UCY|50"] is False


def test_gate_allows_honest_no_primary_promotion() -> None:
    payload = {
        "alignment": {"val": {"aligned": True}, "test": {"aligned": True}},
        "candidate_count": 5,
        "test_eval": {
            "metric_vs_endpoint_ade": {
                "all_improvement": 0.01,
                "hard_failure_improvement": 0.02,
                "easy_degradation": 0.0,
            },
            "near_collision_005_delta_vs_endpoint": -0.001,
        },
        "comparison_to_stage42_cq": {"delta_vs_cq": {"all_improvement": -0.001}},
        "deployment_decision": {"promote_full_waypoint_as_primary_deployable_dynamics": False},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = df._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_df_all_hard_proximity_repair_pass_no_primary_promotion"


def test_run_writes_isolated_outputs(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle()

    def fake_load_split(split: str) -> dict:
        return bundle

    def fake_eval(endpoint, full, keys, choices, min_sep, margin, **kwargs):
        use_full = np.array([bool(choices)] * 8)
        metric = {
            "rows": 8,
            "all_improvement": 0.05 if any(choices.values()) else 0.0,
            "t50_improvement": 0.04 if any(choices.values()) else 0.0,
            "t100_raw_frame_diagnostic_improvement": 0.02 if any(choices.values()) else 0.0,
            "hard_failure_improvement": 0.06 if any(choices.values()) else 0.0,
            "easy_degradation": 0.0,
            "switch_rate": float(np.mean(use_full)),
        }
        return {
            "selected_xy": np.ones((8, 4, 2)),
            "selected_ade": np.ones(8) * 0.95,
            "selected_fde": np.ones(8) * 0.95,
            "use_full": use_full,
            "guarded_off": 0,
            "metric_vs_endpoint_ade": metric,
            "metric_vs_floor_ade": metric,
            "metric_vs_endpoint_fde": metric,
            "metric_vs_floor_fde": metric,
            "near_collision_005_delta_vs_endpoint": -0.001,
            "p05_min_distance_delta_vs_endpoint": 0.0,
            "joint_safety": {"composer_minus_endpoint": {"near_collision_rate_005_delta": -0.001}},
        }

    monkeypatch.setattr(df, "_load_split", fake_load_split)
    monkeypatch.setattr(df, "_evaluate_guarded", fake_eval)
    monkeypatch.setattr(df, "MIN_SLICE_ROWS", 1)
    monkeypatch.setattr(
        df,
        "_compare_to_cq",
        lambda test_eval: {
            "cq_source": "unit",
            "cq_metric_vs_endpoint_ade": {},
            "delta_vs_cq": {
                "all_improvement": 0.01,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.0,
                "hard_failure_improvement": 0.01,
                "easy_degradation": 0.0,
                "switch_rate": 0.0,
                "near_collision_005_delta_vs_endpoint": -0.001,
            },
        },
    )
    monkeypatch.setattr(df, "REPORT_JSON", tmp_path / "full_waypoint_all_hard_proximity_repair_stage42.json")
    monkeypatch.setattr(df, "REPORT_MD", tmp_path / "full_waypoint_all_hard_proximity_repair_stage42.md")
    monkeypatch.setattr(df, "REPORT_CSV", tmp_path / "full_waypoint_all_hard_proximity_repair_stage42.csv")
    monkeypatch.setattr(df, "GATE_MD", tmp_path / "stage42_stage_df_gate.md")
    result = df.run_stage42_full_waypoint_all_hard_proximity_repair(refresh_readmes=False)
    assert result["stage42_df_gate"]["passed"] == result["stage42_df_gate"]["total"]
    assert df.REPORT_JSON.exists()
    assert df.REPORT_MD.exists()
    assert df.REPORT_CSV.exists()
