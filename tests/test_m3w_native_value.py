import numpy as np

from src.world_model import m3w_native_value as model
from src.world_model import m3w_query_risk_critic as previous


def test_target_preserves_original_completion_penalty_and_margin():
    z = np.zeros((3, 12, 2))
    y = z + np.array([1., 2.])[None, None]
    p = y.copy()
    p[0] += 4.
    mask = np.ones((3, 12), bool)
    mask[-1] = False
    v = model.action_targets(z, p, y, mask, 3.)
    old = previous.action_targets(z, p, y, mask, 3.)
    np.testing.assert_allclose(v[:, 0], old[:, 0])
    np.testing.assert_allclose(v[:, 1:3]-.02*v[:, 3:], old[:, 1:3])
    assert v[-1, 3] == v[-1, 4] == 0.


def test_score_penalizes_harm_magnitude_not_reference_positivity_alone():
    pred = np.array([[4., 3., 2., 10., 5.], [1., 0., 0., 10., 5.]])
    scores = model.score(pred)
    assert scores[:, 0].argmax() == 0
    assert scores[:, 1].argmax() == 1
