import numpy as np

from src import stage42_fe_source_robustness_audit as fg


def test_source_name_keeps_last_path_components() -> None:
    assert fg._source_name("/a/b/c/d/e.txt") == "b/c/d/e.txt"
    assert fg._source_name("short.txt") == "short.txt"


def test_metric_for_empty_mask_returns_zero_row_metric() -> None:
    data = {
        "horizon": np.asarray([50, 100]),
        "hard": np.asarray([True, False]),
        "failure": np.asarray([False, False]),
        "easy": np.asarray([False, True]),
    }
    ids = np.asarray([0, 1])
    selected = np.asarray([1.0, 1.0])
    floor = np.asarray([2.0, 2.0])
    switch = np.asarray([True, False])

    metric = fg._metric_for_mask(data, ids, selected, floor, switch, np.asarray([False, False]))

    assert metric["rows"] == 0
    assert metric["all_improvement"] == 0.0


def test_gate_passes_with_reported_weak_slices_and_no_overclaim() -> None:
    payload = {
        "source": fg.SOURCE,
        "fe_policy": {"ff_verdict": "stage42_ff_fe_policy_freeze_replay_pass"},
        "summary": {
            "domain_count": 2,
            "source_count": 3,
            "domain_horizon_count": 6,
            "robust_domains": ["UCY", "TrajNet"],
            "weak_domain_horizons": ["UCY|100"],
            "weak_sources": ["weak/source.txt"],
        },
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
            "metric_or_seconds_claim": False,
            "broad_uniform_source_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fg._gate(payload)

    assert gate["verdict"] == "stage42_fg_fe_source_robustness_pass"
    assert gate["gates"]["broad_source_claim_only_if_no_weak_sources"] is True


def test_gate_blocks_broad_source_overclaim_when_weak_sources_exist() -> None:
    payload = {
        "source": fg.SOURCE,
        "fe_policy": {"ff_verdict": "stage42_ff_fe_policy_freeze_replay_pass"},
        "summary": {
            "domain_count": 2,
            "source_count": 3,
            "domain_horizon_count": 6,
            "robust_domains": ["UCY", "TrajNet"],
            "weak_domain_horizons": ["UCY|100"],
            "weak_sources": ["weak/source.txt"],
        },
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
            "metric_or_seconds_claim": False,
            "broad_uniform_source_claim": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fg._gate(payload)

    assert gate["verdict"] == "stage42_fg_fe_source_robustness_partial"
    assert gate["gates"]["broad_source_claim_only_if_no_weak_sources"] is False
