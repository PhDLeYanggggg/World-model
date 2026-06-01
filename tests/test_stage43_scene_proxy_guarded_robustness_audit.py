from __future__ import annotations

import numpy as np

from src import stage43_scene_proxy_guarded_robustness_audit as ad


def test_compact_rows_preserves_caveats() -> None:
    rows = [
        {
            "name": "domain:UCY",
            "rows": 10,
            "ac": {
                "full_waypoint_ade_improvement_vs_floor": 0.2,
                "t50_full_waypoint_ade_improvement_vs_floor": 0.3,
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.1,
                "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.4,
                "easy_degradation_vs_floor": 0.0,
            },
            "delta_vs_stage43_m": {"full_waypoint_ade_improvement_vs_floor": 0.1},
            "scene_proxy_override_rate": 0.8,
            "caveat_reason": "none",
        }
    ]
    compact = ad._compact_rows(rows)
    assert compact[0]["name"] == "domain:UCY"
    assert compact[0]["ac_all"] == 0.2
    assert compact[0]["caveat"] == "none"


def test_gate_passes_caveated_audit_without_all_domain_success() -> None:
    payload = {
        "source": ad.SOURCE,
        "stage43_ac_verdict": "stage43_ac_guarded_scene_proxy_latent_candidate",
        "result_source": "fresh_replay_stage43_ac_slice_robustness_audit",
        "rows": 100,
        "domain_table": [{"rows": 100}],
        "horizon_table": [{"rows": 25}, {"rows": 25}, {"rows": 25}, {"rows": 25}],
        "source_table": [{"rows": 100}],
        "overall": {
            "stage43_ac": {"easy_degradation_vs_floor": 0.0},
            "delta_vs_stage43_m": {
                "full_waypoint_ade_improvement_vs_floor": 0.1,
                "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
                "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            },
        },
        "claim_boundary": {"t100_raw_frame_diagnostic_only": True, "metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "powered_domain_count": 2,
        "positive_powered_domain_count": 1,
        "weak_or_caveat_slices": [{"name": "domain:weak"}],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "scene_proxy_train_only": True,
        },
    }
    gate = ad._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["all_powered_domains_positive"] is False
    assert gate["verdict"] == "stage43_ad_guarded_scene_proxy_caveated_audit_pass"


def test_gate_marks_all_powered_domains_positive() -> None:
    payload = {
        "source": ad.SOURCE,
        "stage43_ac_verdict": "stage43_ac_guarded_scene_proxy_latent_candidate",
        "result_source": "fresh_replay_stage43_ac_slice_robustness_audit",
        "rows": 100,
        "domain_table": [{"rows": 100}],
        "horizon_table": [{"rows": 25}, {"rows": 25}, {"rows": 25}, {"rows": 25}],
        "source_table": [{"rows": 100}],
        "overall": {
            "stage43_ac": {"easy_degradation_vs_floor": 0.0},
            "delta_vs_stage43_m": {
                "full_waypoint_ade_improvement_vs_floor": 0.1,
                "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
                "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            },
        },
        "claim_boundary": {"t100_raw_frame_diagnostic_only": True, "metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "powered_domain_count": 2,
        "positive_powered_domain_count": 2,
        "weak_or_caveat_slices": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "scene_proxy_train_only": True,
        },
    }
    gate = ad._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["all_powered_domains_positive"] is True
    assert gate["verdict"] == "stage43_ad_guarded_scene_proxy_robust_with_caveats"


def test_empty_slice_row_records_caveat() -> None:
    class Tiny:
        pass

    ds = Tiny()
    ds.x = np.zeros((3, 1), dtype=np.float32)
    mask = np.zeros(3, dtype=bool)
    row = ad._slice_row(
        name="empty",
        mask=mask,
        ds=ds,  # type: ignore[arg-type]
        selected_ade=np.ones(3, dtype=np.float32),
        selected_fde=np.ones(3, dtype=np.float32),
        switched=np.zeros(3, dtype=bool),
        ab_allowed=np.zeros(3, dtype=bool),
        stage43_m_metrics={},
        stage43_ab_metrics={},
    )
    assert row["weak_or_caveat"] is True
    assert row["caveat_reason"] == "empty_slice"
