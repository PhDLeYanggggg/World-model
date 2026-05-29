from __future__ import annotations

import numpy as np

from src.stage43_source_level_latent_model import build_source_level_datasets


def test_stage43_source_level_latent_datasets_cover_all_test_domains():
    train, val, test, manifest = build_source_level_datasets(max_train=256, max_val=256, max_test=512, seed=443)
    assert manifest["stage43_f_gate"]["verdict"] == "stage43_f_source_level_split_ready"
    assert train.x.shape[0] == 256
    assert val.x.shape[0] == 256
    assert test.x.shape[0] == 512
    assert set(np.unique(test.domain.astype(str)).tolist()) >= {"ETH_UCY", "TrajNet", "UCY"}
    assert train.x.shape[1] == val.x.shape[1] == test.x.shape[1]


def test_stage43_source_level_latent_inputs_are_causal_shape_only():
    train, _, _, _ = build_source_level_datasets(max_train=128, max_val=128, max_test=128, seed=451)
    forbidden = {
        "future_endpoint_x",
        "future_endpoint_y",
        "future_waypoint",
        "central_velocity",
        "test_endpoint_goal",
    }
    joined = " ".join(train.feature_names)
    assert not any(name in joined for name in forbidden)
    assert "history_dx_tail0" in train.feature_names
    assert "baseline_pred_rel_0" in train.feature_names
