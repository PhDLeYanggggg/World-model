import numpy as np

from src import stage42_safety_aware_joint_objective_training as fd


def test_blend_teacher_targets_only_changes_masked_rows() -> None:
    labels = np.zeros((3, 2, 2), dtype=np.float32)
    teacher = np.ones((3, 2, 2), dtype=np.float32) * 4.0
    mask = np.asarray([True, False, True])

    out = fd._blend_teacher_targets(labels, teacher, mask, alpha=0.25)

    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 0.0)
    assert np.allclose(out[2], 1.0)


def test_safety_mask_modes_are_boolean_and_train_quantile() -> None:
    signals = {
        "future_close_008": np.asarray([0.0, 1.0, 0.0, 0.0]),
        "future_close_005": np.asarray([0.0, 0.0, 1.0, 0.0]),
        "base_unsafe": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "base_close_008": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "risk": np.asarray([0.1, 0.5, 1.0, 2.0]),
    }
    train = np.asarray([True, True, True, False])

    loose = fd._safety_mask(signals, train, "future_or_base_unsafe")
    strict = fd._safety_mask(signals, train, "strict_near005")
    topq = fd._safety_mask(signals, train, "risk_top_quartile")

    assert loose.dtype == bool
    assert loose.tolist() == [True, True, False, True]
    assert strict.tolist() == [False, False, True, True]
    assert topq.tolist() == [False, False, True, True]


def test_gate_keeps_stage5c_and_smc_false() -> None:
    payload = {
        "source": fd.SOURCE,
        "split_stats": {"by_split": {"test": {"rows": 10}}},
        "label_stats": {"test_full_waypoint_rows": 10},
        "teacher_regularizer": {"teacher_source": "Stage42-FA waypointwise group repel"},
        "model": {
            "candidate_count": 12,
            "selected": {"val_score": 1.0},
            "metrics": {
                "protected_selected_candidate": {
                    "all_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                }
            },
        },
        "comparison_to_prior": {
            "delta_vs_stage42_di": {"all_improvement": 0.0, "hard_failure_improvement": 0.0},
            "delta_vs_stage42_fc": {"all_improvement": 0.0, "hard_failure_improvement": 0.0},
            "near_delta_vs_stage42_di": 0.0,
            "near_delta_vs_stage42_fc": -0.1,
        },
        "no_leakage": {
            "future_waypoint_labels_loss_only": True,
            "teacher_regularizer_train_loss_only": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "validation_only_model_selection": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fd._gate(payload)

    assert gate["gates"]["stage5c_false"] is True
    assert gate["gates"]["smc_false"] is True
    assert gate["gates"]["no_future_inference_input"] is True
