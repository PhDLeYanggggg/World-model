import numpy as np
import pytest

from src.world_model.m3w_matched_path_choice import features, select, targets


def test_action_descriptor_is_rigid_and_scale_invariant():
    rng = np.random.default_rng(5)
    h = rng.normal(size=(4, 8, 2)).cumsum(1)
    cv = np.arange(1, 13)[None, :, None]*(h[:, -1:]-h[:, -2:-1])
    delta = rng.normal(size=(4, 6, 12, 2))
    delta /= np.linalg.norm(delta, axis=-1).max(2)[:, :, None, None]
    bank = cv[:, None]+delta
    x, _ = features(h, cv, bank)
    rot = np.array([[0., -1.], [1., 0.]])
    other, _ = features(17*h @ rot+10, 17*cv @ rot, 17*bank @ rot)
    np.testing.assert_allclose(x, other, atol=1e-7)
    with pytest.raises(ValueError, match="common positive radius"):
        features(h, cv, cv[:, None]+delta*np.arange(1, 7)[None, :, None, None])


def test_missing_steps_debit_gain_and_easy_penalty_abstains_to_anchor():
    cv = np.zeros((1, 12, 2))
    bank = cv[:, None].repeat(2, 1)
    bank[:, 0, :, 0] = 1.
    bank[:, 1, :, 0] = -1.
    future = cv.copy()
    future[:, :6, 0] = 1.
    future[:, 6:] = np.nan
    scores = targets(cv, bank, np.ones((1, 2)), future, np.isfinite(future).all(2), 1.)
    np.testing.assert_array_equal(scores, [[[0., 0.], [-1., 1.]]])
    predicted = np.array([[[.5, .6], [.1, 0.]], [[.5, 0.], [.1, 0.]]])
    assert select(predicted, fallback_index=1)["agreement"].tolist() == [1, 0]
