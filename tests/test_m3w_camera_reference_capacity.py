import numpy as np

from src.evaluation import m3w_camera_reference_capacity as m
from src.evaluation.m3w_cv_anchor_budget import candidate_terms


def test_linear_menu_keeps_constant_motion_and_separates_last_point_noise():
    h = np.zeros((2, 8, 2))
    h[..., 0] = np.arange(8)
    h[1, -1, 0] += 4.
    p = m.linear_menu(h, 12)
    np.testing.assert_allclose(p[0, :, :, 0], np.broadcast_to(np.arange(1, 13), (4, 12)))
    assert not np.allclose(p[1, 0], p[1, 1])
    np.testing.assert_allclose(p[1, 1, 1]-p[1, 1, 0], p[1, 0, 1]-p[1, 0, 0])
    np.testing.assert_allclose(p, m.linear_menu(h+53., 12))


def test_rank_keeps_fixed_count_and_never_amplifies():
    h = np.zeros((3, 8, 2))
    h[..., 0] = np.arange(8)
    h[:, -1, 0] += [1., 3., 4.]
    cv = (h[:, -1:]-h[:, -2:-1])*np.arange(1., 13.)[None, :, None]
    raw = m.linear_menu(h, 12)
    paths, legal = m.matched_menu(cv, raw, np.array([False, True, True]), np.full(3, .5))
    np.testing.assert_array_equal(paths[0], np.broadcast_to(cv[0], paths[0].shape))
    rank = m.past_rank(h, cv, paths, legal, np.zeros(3, dtype=int), [1])
    edits = np.linalg.norm(rank-cv, axis=-1).max(1)
    assert (edits>0.).sum() == 1
    np.testing.assert_allclose(edits.sum(), .5)


def test_radius_bound_dominates_random_finite_paths_with_missing_steps():
    rng = np.random.default_rng(50)
    cv, ols, y = rng.normal(size=(3, 5, 3, 2))
    valid = rng.random((5, 3)) > .25
    q = np.array([0, 0, 0, 1, 1])
    eligible = np.array([True, False, True, True, True])
    radius = np.array([1., 0., 1., .2, .2])
    upper = m.radius_bounds(cv, ols, y, valid, q, eligible, radius, [1, 1])
    for _ in range(100):
        change = [rng.choice([0, 2]), rng.choice([3, 4])]
        direction = rng.normal(size=(2, 3, 2))
        delta = direction/np.linalg.norm(direction, axis=2, keepdims=True)
        p = cv.copy()
        p[change] += delta*radius[change, None, None]
        w = 1./np.bincount(q)[q]
        mass, base, gain, _ = candidate_terms(cv, ols, p[:, None], y, valid)
        effect = 100.*(w @ gain[:, 0]+np.array([0., w@base]))/(w@mass)
        assert np.all(effect <= np.asarray(upper['conservative_pp'])+1e-8)
