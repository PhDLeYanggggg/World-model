import numpy as np
import pytest

from src.world_model.m3w_past_sampling import SCHEDULES, candidates, fit_velocity
from src.evaluation.m3w_sampling_risk_bound import lower_bound
from src.evaluation.m3w_sampling_convex_bound import hull_distance


def test_linear_tracks_agree_for_all_schedules_in_native_units():
    velocity = np.array([[.5, -1.], [2., 3.]])
    h = 112+np.arange(-7., 1.)[None, :, None]*velocity[:, None]
    for t in SCHEDULES.values():
        np.testing.assert_allclose(fit_velocity(h, t), velocity, atol=1e-13)
    with pytest.raises(ValueError, match="past offsets"):
        fit_velocity(h, [-4, -2, 1])


def test_changed_history_cannot_intervene_outside_frozen_recipients():
    h = np.zeros((2, 8, 2))
    h[:, :, 0] = np.arange(8)**2
    cv = np.zeros((2, 12, 2))
    ols = cv+2.
    paths = candidates(h, cv, ols, np.array([True, False]))
    assert set(paths) == set(SCHEDULES)
    assert paths["recent"][0, 0, 0] == 1.
    assert paths["wide_uniform"][0, -1, 0] == pytest.approx(48.)
    for p in paths.values():
        np.testing.assert_array_equal(p[1], cv[1])


def test_hindsight_lower_bound_keeps_unknown_debit_and_reference_support():
    b = np.array([2., 10., np.nan])
    e = np.array([[2.2, 2.1], [9., 11.], [np.nan, np.nan]])
    d = np.array([[1., .5], [1., 2.], [4., 1.]])
    choice, bound = lower_bound(e, d, b, np.array([True, True, False]), np.ones(3, bool), 4.)
    np.testing.assert_array_equal(choice, [1, 0, 1])
    assert bound["all_lower"] == pytest.approx(1.1/12)
    assert bound["easy_lower"] == pytest.approx(1.1/2)
    assert bound["signed_easy_lower"] == pytest.approx(.1/2)


def test_convex_hull_distance_handles_interior_and_degenerate_bank():
    q = np.array([[[.5, .5], [2., .5]], [[.5, .5], [2., .5]]])
    v = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                  [[0., 0.], [1., 0.], [1., 0.], [1., 0.]]])
    np.testing.assert_allclose(hull_distance(q, v), [[0., 1.], [.5, np.sqrt(1.25)]])
