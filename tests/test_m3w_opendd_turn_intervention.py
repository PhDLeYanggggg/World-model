import numpy as np
import pytest

from scripts.run_m3w_opendd_turn_intervention import append_turn, pin_singletons


def test_turn_adds_only_past_derived_candidate():
    angles = np.arange(7) * .12
    step = np.column_stack((np.cos(angles), np.sin(angles)))
    curved = np.vstack(([0., 0.], step.cumsum(0)))
    h = np.stack((curved, np.zeros_like(curved)))
    bank = np.ones((2, 3, 12, 2))
    new = append_turn(h, bank)
    np.testing.assert_array_equal(new[:, :3], bank)
    np.testing.assert_array_equal(new[1, -1], 0.)
    assert np.linalg.norm(np.diff(new[0, -1], n=2, axis=0)) > 0.
    rotation = np.array([[0., 1.], [-1., 0.]])
    rotated = append_turn(2. * (h @ rotation) + 100., 2. * (bank @ rotation))
    np.testing.assert_allclose(rotated, 2. * (new @ rotation), atol=1e-12)


def test_pinning_preserves_singletons_and_rejects_different_slot_radii():
    q = np.array([0, 0, 1, 1, 1, 2, 2])
    cv = np.zeros((7, 12, 2))
    joint = cv.copy()
    joint[[0, 2, 3], :, 0] = np.array([.5, 1., 2.])[:, None]
    additive = joint.copy()
    additive[[0, 1]] = joint[[1, 0]]
    predicted = joint.copy()
    predicted[[3, 4]] = joint[[4, 3]]
    double = pin_singletons(cv, q, additive, joint, {"candidate": predicted})
    np.testing.assert_array_equal(double, [False, False, True, True, True, False, False])
    np.testing.assert_array_equal(predicted[:2], additive[:2])
    np.testing.assert_array_equal(predicted[2:5], joint[[2, 4, 3]])
    predicted[4, :, 0] += .1
    with pytest.raises(AssertionError):
        pin_singletons(cv, q, additive, joint, {"candidate": predicted})
