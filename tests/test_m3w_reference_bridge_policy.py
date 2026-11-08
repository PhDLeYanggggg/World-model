import numpy as np

from src.world_model.m3w_reference_bridge_policy import query_choices


def test_joint_can_compensate_risk_without_double_selecting_an_agent():
    cost = np.zeros((3, 3, 6))
    legal = np.zeros((3, 3), dtype=bool)
    legal[:2, 0] = True
    cost[0, 0] = [3, 3, .2, .2, 1, 1]
    cost[1, 0] = [2, 2, 0, 0, 10, 10]
    c = query_choices(cost, legal, np.array([1, 1, 1]), np.arange(3))
    assert c["joint"].tolist() == [0, 0, -1]
    assert c["matched_independent"].tolist() == [0, 0, -1]


def test_single_query_does_not_convert_an_identity_to_a_safety_pass():
    cost = np.ones((1, 2, 6))
    out = query_choices(cost, np.ones((1, 2), bool), np.ones(1), np.zeros(1))
    assert all(v.tolist() == [-1] for v in out.values())
