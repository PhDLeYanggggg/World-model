import numpy as np
import pytest

from src.evaluation.m3w_pair_annotation_budget import aggregate, curves, minimum_relaxation


def test_two_missing_endpoints_must_both_be_reviewed():
    x, w = aggregate(np.array([[10, 11], [11, 10], [12, -1]]), np.array([3., 3., 2.]))
    out = curves(x, w)
    np.testing.assert_allclose(out['chronological'], [0., 0., .75, 1.])
    np.testing.assert_allclose(out['incident_mass'], [0., 0., .75, 1.])
    np.testing.assert_allclose(out['dependency_greedy'], [0., 0., .75, 1.])
    assert minimum_relaxation(x, w, .75) == pytest.approx(2.)
    assert minimum_relaxation(x, w, 1.) == pytest.approx(3.)


def test_singletons_release_mass_at_exact_budget():
    x, w = aggregate(np.array([[1, -1], [8, -1], [10, 8]]), np.array([1., 4., 3.]))
    out = curves(x, w)
    np.testing.assert_allclose(out['dependency_greedy'], [0., .5, .875, 1.])
    assert minimum_relaxation(x, w, .5) == pytest.approx(1.)
