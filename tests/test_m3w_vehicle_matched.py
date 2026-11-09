import numpy as np

from src.world_model import m3w_vehicle_matched as model


def test_joint_combines_mixed_family_actions_and_controls_match_exact_radius():
    cv = np.zeros((3, 12, 2))
    raw = np.empty((3, 2, 12, 2))
    raw[:, 0] = [4., 0.]
    raw[:, 1] = [0., 4.]
    paths = model.same_radius(cv, raw, np.array([0, 3]))
    x = np.zeros((3, 2, 6))
    x[..., :2] = 1.
    x[..., 2:4] = 2.
    x[..., 4:] = 50.
    x[0, 0, :4] = [4., 4., .9, .9]
    x[1, 1, :4] = [4., 4., .9, .9]
    x[2, 0, :4] = [7., 7., 5., 5.]
    pred = model.choose(cv, paths, x, np.array([0, 3]))
    for a in pred.values():
        np.testing.assert_allclose(np.sort(np.linalg.norm(a-cv, axis=-1).max(axis=-1)), [0., 1., 1.])
    assert pred['joint'][2, 0, 0] == 0.
    assert pred['matched_independent'][2, 0, 0] == 1.
    assert np.count_nonzero(pred['matched_family'][:, 0, 0]) == 2


def test_negative_predicted_gain_forces_all_controls_to_fallback():
    x = np.ones((3, 2, 6))
    x[..., :2] = -1.
    for a in model.decide(x, np.ones((3, 2), dtype=bool), max_edits=2).values():
        assert a.shape == (0, 2)


def test_uniform_unavailable_cannot_be_treated_as_a_two_edit_win():
    legal = np.array([[True, False], [False, True], [False, False]])
    x = np.ones((3, 2, 6))
    x[..., 2:4] = 0.
    assert all(len(a) == 1 for a in model.decide(x, legal, max_edits=2).values())


def test_hindsight_requires_a_real_easy_reference():
    from scripts.diagnose_m3w_vehicle_matched import admission

    v = np.array([[1., 1., .01, 0., 1., 0.], [1., 1., .01, 0., 1., .5]])
    np.testing.assert_array_equal(admission(v, require_easy=False)[-1], [True, True])
    np.testing.assert_array_equal(admission(v, require_easy=True)[-1], [False, True])
