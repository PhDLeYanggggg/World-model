import numpy as np

from src.world_model.m3w_recording_risk_pool import group_offsets, prediction, query_menu, select


def test_risk_price_can_change_actor_at_exact_count():
    paths = np.zeros((2, 5, 12, 2))
    paths[..., 0] = np.arange(5)[None, :, None]
    v = np.zeros((2, 4, 3))
    v[0, 1] = [2., 2., 2.]
    v[1, 1] = [1., 0., 0.]
    x, mass = query_menu(paths, v, np.ones((2, 2)), np.array([1, 0]))
    np.testing.assert_array_equal(np.count_nonzero(x, axis=0), 1)
    np.testing.assert_array_equal(x[:, 0], [1, 0])
    np.testing.assert_array_equal(x[:, 1], [0, 1])
    risk, gain = prediction(mass)
    np.testing.assert_array_equal(risk[1], [0., 0.])
    assert gain[0] == 1.
    assert gain[1] == .5
    assert select(risk, gain, np.zeros_like(risk)) == 1


def test_offsets_leave_out_the_entire_source_and_weight_groups():
    sources = {"held": np.full((2, 4, 2), 100.), "small": np.full((1, 4, 2), 2.),
               "large": np.full((20, 4, 2), 4.)}
    raw, mid, upper = group_offsets(sources, "held")
    np.testing.assert_array_equal(raw, 0.)
    np.testing.assert_array_equal(mid, 3.)
    np.testing.assert_array_equal(upper, 4.)


def test_abstention_if_support_missing_or_gain_nonpositive():
    risk = np.array([[.01, np.inf], [.0, .0], [.021, .0], [.0, .0]])
    assert select(risk, np.array([1., 0., 1., -1.]), np.zeros_like(risk)) is None
