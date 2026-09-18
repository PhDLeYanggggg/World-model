import numpy as np
import pytest

from src.evaluation.m3w_recording_diagnostic import (
    recording_resamples, paired_gain_interval, error_summary,
)


def test_block_draws_deterministic_and_paired():
    groups = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
    labels, inv, counts = recording_resamples(groups)
    assert counts.shape == (2000, 3)
    np.testing.assert_array_equal(counts.sum(1), 3)
    np.testing.assert_array_equal(counts, recording_resamples(groups)[2])
    cv = np.arange(1, 7, dtype=float)
    result = paired_gain_interval(cv, cv*.8, cv, inv, counts)
    np.testing.assert_allclose(result['point_percent'], 20)
    np.testing.assert_allclose(result['conditional_recording_ci95'], [20, 20])
    identical = paired_gain_interval(cv, cv, cv, inv, counts)
    assert identical['conditional_recording_ci95'] == [0, 0]


def test_duplicate_windows_do_not_inflate_block_sample_size():
    groups = np.array(['a', 'a', 'b', 'c'])
    cv = np.array([1., 2., 3., 8.])
    pred = np.array([2., 1., 2., 5.])
    _, inv, counts = recording_resamples(groups)
    a = paired_gain_interval(cv, pred, cv, inv, counts)
    _, inv2, counts2 = recording_resamples(np.repeat(groups, 5))
    b = paired_gain_interval(np.repeat(cv, 5), np.repeat(pred, 5), np.repeat(cv, 5), inv2, counts2)
    np.testing.assert_allclose(a['conditional_recording_ci95'], b['conditional_recording_ci95'])
    assert a['point_percent'] == b['point_percent'] and a['records'] == b['records'] == 3


def test_window_weighting_is_not_recording_mean():
    _, inv, counts = recording_resamples(np.array(['a', 'a', 'a', 'b']))
    a = paired_gain_interval(np.ones(4), np.array([0., 0., 0., 1.]), np.ones(4), inv, counts)
    assert a['point_percent'] == 75


def test_zero_error_subset_is_not_a_passing_percentage():
    _, inv, counts = recording_resamples(np.array(['a', 'b']))
    result = paired_gain_interval(np.zeros(2), np.ones(2), np.zeros(2), inv, counts)
    assert result['point_percent'] is None and result['conditional_recording_ci95'] is None
    assert result['valid_resamples'] == 0 and result['invalid_zero_denominator_resamples'] == 2000
    summary = error_summary(np.array([2., 1.]), np.array([2., 1.]), np.array([0., 2.]),
                            np.array([.001, .001]), np.ones(2), np.array([False, True]))
    assert summary['easy_percentage_degradation'] is None
    assert summary['easy_pixel_harm'] == .002 and summary['gain_percent'] == -50


def test_blocks_and_losses_reject_bad_alignment():
    with pytest.raises(ValueError):
        recording_resamples(['a', 'a'])
    _, inv, counts = recording_resamples(['a', 'b'])
    with pytest.raises(ValueError):
        paired_gain_interval([1., np.nan], [1., 1.], [1., 1.], inv, counts)
    with pytest.raises(ValueError):
        paired_gain_interval([1., 1.], [1., 1.], [1., 1.], inv, counts, mask=np.ones(2))
