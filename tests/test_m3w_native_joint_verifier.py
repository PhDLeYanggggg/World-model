import numpy as np

from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics


def test_explicit_distances_keep_unknowns_and_exact_zero():
    y = np.array([[[0., 0.], [3., 4.]], [[np.nan, np.nan], [3., 4.]],
                  [[np.nan, np.nan], [np.nan, np.nan]]])
    p = np.array([[[0., 0.], [3., 4.]], [[8., 8.], [0., 0.]], [[0., 0.], [0., 0.]]])
    mask = np.array([[True, True], [False, True], [False, False]])
    ade, fde = distances(p, y, mask, np.array([2., 3., 1.]))
    np.testing.assert_array_equal(ade, [0., 15., np.nan])
    np.testing.assert_array_equal(fde, [0., 15., np.nan])


def test_explicit_distances_agree_on_partial_rollouts():
    rng = np.random.default_rng(401)
    p, y = rng.normal(size=(2, 101, 12, 2))
    mask = rng.random((101, 12)) > .2
    scale = rng.uniform(.01, 100, 101)
    for x, z in zip(distances(p, y, mask, scale), native_errors(p, y, mask, scale)):
        np.testing.assert_allclose(x, z, rtol=1e-14, atol=1e-14)


def test_separate_scene_and_bootstrap_reductions():
    sites = np.array(['a', 'b', 'a', 'b', 'a', 'b'])
    model, reference = np.array([1., 2., 3., 2., np.nan, 4.]), np.array([2., 1., 2., 4., np.nan, 5.])
    expected = paired_scene_metrics(model, reference, sites, expected_scenes=['a', 'b'],
        dataset='synthetic', coordinate_unit='local', bootstrap_resamples=3000)
    assert check_reduction(model, reference, sites, ['a', 'b'], expected) == 2


def test_zero_reference_does_not_become_percentage():
    model, reference = np.array([0., .001]), np.zeros(2)
    sites = np.array(['a', 'b'])
    expected = paired_scene_metrics(model, reference, sites, expected_scenes=['a', 'b'],
        dataset='synthetic', coordinate_unit='local', bootstrap_resamples=3000)
    assert check_reduction(model, reference, sites, ['a', 'b'], expected) == 2
    assert expected['equal_scene_gain_percent'] is None
