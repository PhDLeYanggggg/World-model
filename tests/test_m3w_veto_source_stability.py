import numpy as np

from src.world_model import m3w_veto_source_stability as model


def test_identical_source_models_have_no_penalty():
    moment = (np.ones((1, 1, 1)), np.full((1, 1, 2), -.4))
    point, cautious = model.correction(np.ones((2, 1, 1)), *model.fit([moment] * 4))
    np.testing.assert_allclose(point, cautious)


def test_source_sensitivity_only_removes_negative_revisions():
    x = np.ones((1, 1, 1))
    full = np.array([[[-.3, .2]]])
    deleted = np.array([[[[-.3, 1.]]], [[[-1.1, -1.]]], [[[.1, .2]]]])
    point, cautious = model.correction(x, full, deleted)
    np.testing.assert_allclose(point, [[[-.3, .2]]])
    np.testing.assert_allclose(cautious, [[[0., .2]]])
