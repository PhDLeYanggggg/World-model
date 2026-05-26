import numpy as np

from src import stage42_source_level_full_waypoint_eval as s42am


def _tiny_data():
    return {
        "source_file": np.asarray(["/x/a.txt", "/x/b.txt", "/x/c.txt", "/x/a.txt"], dtype="U32"),
        "dataset": np.asarray(["ETH_UCY", "TrajNet", "UCY", "ETH_UCY"], dtype="U16"),
        "scene_id": np.asarray(["s0", "s1", "s2", "s0"], dtype="U16"),
        "horizon": np.asarray([10, 25, 50, 100], dtype=np.int16),
        "hard": np.asarray([True, False, True, False]),
        "failure": np.asarray([False, False, True, False]),
        "easy": np.asarray([False, True, False, True]),
        "current_x": np.zeros(4, dtype=np.float32),
        "current_y": np.zeros(4, dtype=np.float32),
        "past_start_x": np.zeros(4, dtype=np.float32),
        "past_start_y": np.zeros(4, dtype=np.float32),
        "dt_frame_step": np.ones(4, dtype=np.float32),
        "scale": np.ones(4, dtype=np.float32),
        "history_scalar": np.ones((4, 9), dtype=np.float32),
        "history_seq": np.ones((4, 64, 7), dtype=np.float32),
        "prototype_likelihood": np.ones((4, 8), dtype=np.float32),
        "prototype_entropy": np.ones(4, dtype=np.float32),
        "goal_ambiguity": np.ones(4, dtype=np.float32),
        "family_pred": np.ones((4, 8, 2), dtype=np.float32),
    }


def test_source_split_has_no_group_overlap():
    data = _tiny_data()
    split, group = s42am._split_arrays(data)
    stats = s42am._source_stats(data, split, group)
    assert stats["source_overlap_pass"] is True
    assert set(stats["group_overlap"]) == {"train_val", "train_test", "val_test"}


def test_feature_matrix_excludes_label_only_fields():
    data = _tiny_data()
    floor = {"floor_endpoint": np.zeros((4, 2), dtype=np.float32)}
    features, names = s42am._feature_matrix(data, floor)
    assert features.shape[0] == 4
    assert "family_fde" not in " ".join(names)
    assert "safe_strongest_idx_old" not in " ".join(names)


def test_gate_rejects_metric_stage5c_smc_claims():
    base_metric = {
        "rows": 47458,
        "all_improvement": 0.01,
        "t50_improvement": 0.02,
        "hard_failure_improvement": 0.03,
        "easy_degradation": 0.0,
    }
    result = {
        "split_stats": {
            "by_split": {
                "test": {
                    "rows": 47458,
                    "domains": {"TrajNet": 37918, "UCY": 9540},
                }
            },
            "source_overlap_pass": True,
        },
        "label_stats": {"test_full_waypoint_rows": 100},
        "model": {
            "metrics": {"protected_ridge_source_level": base_metric},
            "policy": {"slices": {}},
            "bootstrap": {
                "all": {"bootstrap_n": 1000},
                "t50": {"bootstrap_n": 1000},
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
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
    gate = s42am._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["positive_transfer"] is True
