import numpy as np
import pytest

from scripts import audit_m3w_av2_incumbent_timing as audit
from src.world_model import m3w_av2_incumbent_timing as model


def test_candidate_profiles_keep_the_same_two_recipients_and_each_peak():
    cv = np.zeros((4, 12, 2))
    a = cv.copy()
    a[0, :, 0] = 1.
    a[2, :, 1] = 2.
    bank, edited, shape = model.paths(cv, a)
    assert len(bank) == 9
    np.testing.assert_array_equal(bank[0], a)
    np.testing.assert_array_equal(edited, np.tile([0, 2], (9, 1)))
    np.testing.assert_array_equal(shape[0], [0, 0])
    np.testing.assert_array_equal(bank[:, [1, 3]], 0.)
    np.testing.assert_allclose(np.linalg.norm(bank, axis=-1).max(-1), np.tile([1., 0., 2., 0.], (9, 1)))


def test_incumbent_deduplicates_endpoint_only_actions():
    cv = np.zeros((3, 12, 2))
    a = cv.copy()
    a[:2, -1, 0] = 1.
    assert len(model.paths(cv, a)[0]) == 1
    a[1] = 0.
    with pytest.raises(ValueError, match="exactly two"):
        model.paths(cv, a)


def test_replay_requires_both_past_trajectory_and_pair_gain():
    x = np.zeros((4, 4, 8))
    x[1, :, 0], x[1, :, 5] = 2., -1.
    x[2, :, 0], x[2, :, 5] = -1., 2.
    x[3, :, 0], x[3, :, 5] = 2., 3.
    np.testing.assert_array_equal(model.replay_score(x.reshape(4, -1)), [0., -1., -1., 2.])


def test_locality_separates_same_actors_from_the_same_assignment():
    fixed, swapped = audit.locality(np.array([[0, 2], [2, 0], [1, 2]]), [.5, 1.], [.5, 0., 1.])
    np.testing.assert_array_equal(fixed, [True, False, False])
    np.testing.assert_array_equal(swapped, [True, True, False])
