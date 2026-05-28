import numpy as np

from src import stage42_source_level_context_repair as ix


def test_feature_masks_include_context_and_baseline_controls():
    names = [
        "history_scalar_0",
        "history_scalar_1",
        "prototype_0",
        "prototype_entropy",
        "goal_ambiguity",
        "safe_baseline_rel_0",
        "family_baseline_rel_0",
        "floor_rel_x",
        "domain_UCY",
        "horizon_50",
    ]
    masks = ix._feature_masks(names)
    assert masks["full"].sum() == len(names)
    assert masks["baseline_family"][5]
    assert masks["baseline_family"][8]
    assert masks["context_only"][0]
    assert masks["context_only"][2]
    assert masks["goal_neighbor_context"][2]
    assert not masks["goal_neighbor_context"][0]


def test_sample_weights_emphasize_t50_and_hard_failure():
    data = {
        "horizon": np.asarray([10, 50, 100, 50]),
        "hard": np.asarray([False, False, True, True]),
        "failure": np.asarray([False, False, False, True]),
    }
    w = ix._sample_weights(data, "t50_hard_t100")
    assert w[0] == 1.0
    assert w[1] > w[0]
    assert w[2] > w[0]
    assert w[3] > w[1]


def test_gate_records_completed_negative_context_repair():
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "trials": {t["name"]: {"trial": t} for t in ix.TRIALS},
        "summary": {
            "best_trial_metric": {
                "all_improvement": 0.1,
                "t50_improvement": 0.1,
                "hard_failure_improvement": 0.1,
                "easy_degradation": 0.0,
            },
            "context_claim_verdict": "stage42_ix_context_repair_negative_context_still_not_incremental",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = ix._gate(result)
    assert gate["gates"]["weighted_repair_trials_complete"]
    assert not gate["gates"]["context_incremental_claim_supported"]
    assert gate["verdict"] == "stage42_ix_context_repair_completed_context_not_proven"


def test_stage42_ix_run_completes_trials():
    result = ix.run_stage42_source_level_context_repair(use_cached=True)
    gate = result["stage42_ix_gate"]
    assert gate["gates"]["weighted_repair_trials_complete"]
    assert gate["gates"]["floor_residual_objective_tested"]
    assert gate["gates"]["context_only_trials_tested"]
    assert result["summary"]["context_claim_verdict"] in {
        "stage42_ix_context_repair_positive",
        "stage42_ix_context_repair_negative_context_still_not_incremental",
    }
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
