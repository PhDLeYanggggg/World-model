import numpy as np

from src import stage41_shape_policy_composer as composer


def test_source_name_maps_horizon_families():
    horizon = np.asarray([10, 25, 50, 100], dtype=np.int64)
    policy = {"short": "bridge", "t50": "gain_gate", "t100": "old_shape"}
    names = composer._source_name(policy, horizon)
    assert names.tolist() == ["bridge", "bridge", "gain_gate", "old_shape"]


def test_candidate_policies_cover_all_source_combinations():
    policies = composer._candidate_policies()
    assert len(policies) == 27
    assert {"short": "bridge", "t50": "gain_gate", "t100": "old_shape"} in policies
