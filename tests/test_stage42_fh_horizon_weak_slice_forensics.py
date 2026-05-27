from __future__ import annotations

import numpy as np

from src import stage42_fh_horizon_weak_slice_forensics as fl


def test_corr_handles_constant_and_valid_signal() -> None:
    assert fl._corr(np.ones(5), np.arange(5)) is None
    assert fl._corr(np.arange(5), np.arange(5)) is not None
    assert fl._corr(np.arange(5), np.arange(5)) > 0.99


def test_oracle_summary_reports_headroom_and_margin() -> None:
    data = {
        "dataset": np.asarray(["UCY", "UCY", "UCY"], dtype=object),
        "horizon": np.asarray([50, 50, 50]),
    }
    ids = np.asarray([0, 1, 2])
    evals = {
        "fh": {"selected_ade": np.asarray([8.0, 8.0, 8.0]), "floor_ade": np.asarray([10.0, 10.0, 10.0])},
        "fc": {"selected_ade": np.asarray([5.0, 9.0, 7.0]), "floor_ade": np.asarray([10.0, 10.0, 10.0])},
        "di": {"selected_ade": np.asarray([6.0, 6.0, 9.0]), "floor_ade": np.asarray([10.0, 10.0, 10.0])},
        "floor": {"selected_ade": np.asarray([10.0, 10.0, 10.0]), "floor_ade": np.asarray([10.0, 10.0, 10.0])},
    }

    row = fl._oracle_summary(data, ids, evals, "UCY|50")

    assert row["rows"] == 3
    assert row["oracle_improvement_vs_floor"] > 0
    assert row["oracle_improvement_vs_fh"] > 0
    assert row["best_candidate_distribution"]["fc"] == 2
    assert row["best_candidate_distribution"]["di"] == 1


def test_dominant_root_cause_detects_candidate_level_insufficiency() -> None:
    slice_row = {
        "test_oracle": {
            "rows": 500,
            "oracle_improvement_vs_fh": 0.2,
            "low_margin_share": {"0.05": 0.2},
        },
        "val_oracle": {
            "rows": 500,
            "oracle_improvement_vs_fh": 0.2,
            "low_margin_share": {"0.05": 0.2},
        },
        "test_candidate_table": {
            "fh": {"delta_vs_fh": 0.0},
            "fc": {"delta_vs_fh": -0.1},
        },
        "val_candidate_table": {
            "fh": {"delta_vs_fh": 0.0},
            "fc": {"delta_vs_fh": 0.1},
        },
        "val_feature_signal": {
            "fc": {
                "corr_endpoint_delta_floor_to_gain": 0.01,
                "corr_endpoint_delta_fh_to_gain": 0.02,
                "corr_path_length_to_gain": 0.03,
                "corr_min_distance_to_gain": 0.04,
            }
        },
    }

    reasons = fl._dominant_root_cause(slice_row)

    assert "row_level_switch_required_candidate_level_override_insufficient" in reasons
    assert "validation_to_test_distribution_shift" in reasons
    assert "past_only_proxy_features_weak_for_gain_prediction" in reasons
