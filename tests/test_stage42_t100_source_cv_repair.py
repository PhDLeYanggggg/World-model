import numpy as np

from src import stage42_t100_source_cv_repair as s42ba


def test_t100_train_groups_counts_only_train_t100_rows():
    data = {
        "dataset": np.asarray(["A", "A", "A", "A", "B"]),
        "horizon": np.asarray([100, 50, 100, 100, 100]),
    }
    split = np.asarray(["train", "train", "val", "train", "train"])
    group = np.asarray(["g1", "g1", "g2", "g3", "g4"])
    out = s42ba._t100_train_groups(data, split, group)
    assert out["A"] == [{"group": "g1", "t100_rows": 1}, {"group": "g3", "t100_rows": 1}]
    assert out["B"] == [{"group": "g4", "t100_rows": 1}]


def test_domain_cv_summary_requires_all_safe_positive_folds():
    rows = [
        {"domain": "A", "holdout_h100": {"t100_raw_frame_diagnostic_improvement": 0.1, "easy_degradation": 0.0}, "safe_positive_t100": True},
        {"domain": "A", "holdout_h100": {"t100_raw_frame_diagnostic_improvement": 0.2, "easy_degradation": 0.01}, "safe_positive_t100": True},
        {"domain": "B", "holdout_h100": {"t100_raw_frame_diagnostic_improvement": 0.1, "easy_degradation": 0.0}, "safe_positive_t100": True},
        {"domain": "B", "holdout_h100": {"t100_raw_frame_diagnostic_improvement": -0.1, "easy_degradation": 0.0}, "safe_positive_t100": False},
    ]
    domains = {"A": {"status": "fresh_run"}, "B": {"status": "fresh_run"}, "C": {"status": "not_run", "reason": "blocked"}}
    out = s42ba._domain_cv_summary(rows, domains)
    assert out["A"]["supported_for_t100"] is True
    assert out["B"]["supported_for_t100"] is False
    assert out["C"]["status"] == "not_run"


def test_apply_domain_cv_t100_guard_only_disables_unsupported_domain_h100():
    data = {
        "dataset": np.asarray(["A", "A", "B", "B"]),
        "horizon": np.asarray([100, 50, 100, 25]),
    }
    selected = np.asarray([1.0, 2.0, 3.0, 4.0])
    floor = np.asarray([10.0, 20.0, 30.0, 40.0])
    switch = np.asarray([True, True, True, True])
    support = {
        "A": {"supported_for_t100": True},
        "B": {"supported_for_t100": False},
    }
    out = s42ba._apply_domain_cv_t100_guard(
        data=data,
        selected_ade=selected,
        selected_fde=selected.copy(),
        switch=switch,
        floor_ade=floor,
        floor_fde=floor.copy(),
        domain_support=support,
    )
    assert "A|100" in out["kept_slices"]
    assert "B|100" in out["guarded_slices"]
    assert out["selected_ade"][0] == selected[0]
    assert out["selected_ade"][2] == floor[2]
    assert out["selected_ade"][3] == selected[3]


def test_gate_passes_with_safe_t100_blocker():
    metric = {
        "rows": 100,
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.0,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.5,
    }
    result = {
        "source": "fresh_run",
        "ay_verdict": "stage42_ay_t100_easy_safety_repair_pass",
        "az_verdict": "stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation",
        "source_cv_fold_count": 2,
        "domain_t100_support": {"A": {"supported_for_t100": False}},
        "summary": {
            "supported_t100_domains": [],
            "final_t100_positive": False,
        },
        "final_eval": {
            "guard": {"guarded_slices": {"A|100": {}}},
            "after_cv_guard": {
                "protected": metric,
                "bootstrap": {
                    "all": {"low": 0.01},
                    "t50": {"low": 0.01},
                    "hard_failure": {"low": 0.01},
                    "h100_easy_degradation": {"high": 0.0},
                },
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "final_test_metrics_for_threshold": False,
            "source_cv_from_original_train_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42ba._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ba_t100_source_cv_repair_pass_with_t100_blocker"
