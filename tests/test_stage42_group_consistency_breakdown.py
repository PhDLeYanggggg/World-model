from __future__ import annotations

import numpy as np

from src import stage42_group_consistency_breakdown as hp


def test_local_metric_records_ade_and_fde_improvement() -> None:
    selected_ade = np.asarray([1.0, 2.0, 3.0, 4.0])
    floor_ade = np.asarray([2.0, 2.0, 6.0, 4.0])
    selected_fde = np.asarray([1.0, 4.0, 3.0, 8.0])
    floor_fde = np.asarray([2.0, 8.0, 6.0, 8.0])
    horizon = np.asarray([10, 50, 50, 100])
    hard = np.asarray([False, True, True, False])
    failure = np.asarray([False, False, True, False])
    easy = np.asarray([True, False, False, True])
    switch = np.asarray([False, True, True, False])
    mask = np.ones(4, dtype=bool)
    metric = hp._local_metric(
        selected_ade=selected_ade,
        floor_ade=floor_ade,
        selected_fde=selected_fde,
        floor_fde=floor_fde,
        horizon=horizon,
        hard=hard,
        failure=failure,
        easy=easy,
        switch=switch,
        mask=mask,
    )
    assert metric["rows"] == 4
    assert metric["ade_t50_improvement"] > 0.0
    assert metric["fde_t50_improvement"] > 0.0
    assert metric["switch_rate"] == 0.5


def test_weak_slices_record_non_positive_t50_and_easy_harm() -> None:
    records = [
        {
            "name": "domain:ok",
            "rows": 1000,
            "t50_rows": 100,
            "ade_all_improvement": 0.2,
            "ade_t50_improvement": 0.1,
            "ade_easy_degradation": 0.0,
            "safety": {"near_005_delta_vs_base": 0.0},
        },
        {
            "name": "domain:weak",
            "rows": 1000,
            "t50_rows": 100,
            "ade_all_improvement": 0.1,
            "ade_t50_improvement": 0.0,
            "ade_easy_degradation": 0.05,
            "safety": {"near_005_delta_vs_base": 0.0},
        },
    ]
    weak = hp._weak_slices(records)
    assert len(weak) == 1
    assert weak[0]["name"] == "domain:weak"
    assert "non_positive_t50" in weak[0]["reason"]
    assert "easy_degradation_over_2pct" in weak[0]["reason"]


def test_weak_slices_do_not_mark_t50_when_slice_has_no_t50_rows() -> None:
    records = [
        {
            "name": "horizon:100",
            "rows": 1000,
            "t50_rows": 0,
            "ade_all_improvement": 0.1,
            "ade_t50_improvement": 0.0,
            "ade_easy_degradation": 0.0,
            "safety": {"near_005_delta_vs_base": 0.0},
        }
    ]
    assert hp._weak_slices(records) == []


def test_gate_rejects_metric_or_stage5c_overclaim() -> None:
    row = {
        "name": "all",
        "rows": 47458,
        "ade_all_improvement": 0.2,
        "t10_rows": 1,
        "t25_rows": 1,
        "t50_rows": 1,
        "t100_rows": 1,
        "ade_t10_improvement": 0.2,
        "ade_t25_improvement": 0.2,
        "ade_t50_improvement": 0.2,
        "ade_t100_raw_frame_diagnostic_improvement": 0.2,
        "ade_hard_failure_improvement": 0.2,
        "ade_easy_degradation": 0.0,
        "fde_all_improvement": 0.2,
        "fde_t50_improvement": 0.2,
        "fde_t100_raw_frame_diagnostic_improvement": 0.2,
        "switch_rate": 0.1,
        "harm_over_floor_ade": -1.0,
        "safety": {"near_005_delta_vs_base": 0.0},
    }
    payload = {
        "source": "fresh_run_group_consistency_source_breakdown",
        "policy_artifact": {"hash": "hash"},
        "rows": {"test_rows": 47458},
        "overall": row,
        "breakdown": {
            "by_domain": [{**row, "name": "domain:a"}, {**row, "name": "domain:b"}],
            "by_source": [{**row, "name": "source:a"}, {**row, "name": "source:b"}, {**row, "name": "source:c"}],
            "by_scene": [{**row, "name": "scene:a"}, {**row, "name": "scene:b"}],
            "by_horizon": [
                {**row, "name": "horizon:10"},
                {**row, "name": "horizon:25"},
                {**row, "name": "horizon:50"},
                {**row, "name": "horizon:100"},
            ],
        },
        "weak_slices": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = hp._gate(payload)
    assert gate["gates"]["no_metric_seconds_claim"] is False
    assert gate["passed"] == gate["total"] - 1
