import numpy as np
from src.evaluation.m3w_policy_region_shift import compare_regions
from src.world_model.m3w_multiaction_cost import select_actions


def test_old_teacher_region_can_fail_new_gate_without_being_dropped():
    p = np.tile([1., .001, 2., 1., .001], (3, 7, 1)); p[:, 1, 4] = .5
    y = np.tile([1., .1, 2., 1., .1], (3, 7, 1)); y[-1] = np.nan
    allowed = np.ones((3, 7), bool); current, _ = select_actions(p, allowed, arm='physical_only')
    result = compare_regions(p, y, allowed, current, np.array([1, -1, 1]), np.ones(3), tuple(range(6)))
    assert result['frozen_teacher_region']['action_views'] == 2
    assert result['frozen_teacher_region']['unknown_action_views'] == 1
    assert result['new_interventions_outside_teacher_region'] == 3
    assert result['same_teacher_action_kept'] == 0
