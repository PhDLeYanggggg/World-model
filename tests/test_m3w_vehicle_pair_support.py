import numpy as np
import pytest

from scripts.run_m3w_vehicle_pair_support import fixed_pairs, paired
from src.world_model.m3w_vehicle_pair_support import choose


def test_joint_can_trade_gain_across_references_at_same_radius():
    v = np.zeros((3, 2, 6))
    v[..., :2] = [[[5, 1]], [[5, 1]], [[.1, 5]]]
    out = choose(v, np.ones(3), np.ones((3, 2)))
    np.testing.assert_array_equal(out['independent'][0], [0, 1])
    np.testing.assert_array_equal(out['joint'][0], [0, 2])
    assert len(set(out['uniform'][1])) == 1


def test_requires_matched_family_control_and_equal_radius():
    v = np.zeros((2, 2, 6))
    assert choose(v, np.ones(2), np.eye(2)) == {}
    with pytest.raises(AssertionError):
        choose(v, np.ones(2), np.array([[1., 1.], [1., .5]]))


def test_fixed_pair_keeps_a_third_actor_and_preserves_chronology():
    peak = np.ones((8, 2))
    slots = fixed_pairs(np.array([0, 2, 5, 8]), np.repeat('video', 8), peak)
    assert slots == {'video': (2, 5)}


def test_paired_missing_steps_charge_the_actual_difference():
    a = np.zeros((3, 12, 2))
    b = a.copy()
    b[0, :, 0] = 1.
    d = dict(future=b, valid=np.zeros((3, 12), bool), weight=np.ones(3))
    out = paired(b, a, d, [(0, 3)], 12.)
    assert out['ade_available_pp'] == 0
    assert out['ade_lower_pp'] == -100
    assert out['relative_lower_native'] == pytest.approx(-2/3)
