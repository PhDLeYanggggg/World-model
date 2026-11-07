"""Synthetic checks of the existing row-support versus full-grid estimands."""
import numpy as np
import pytest

from scripts.run_m3w_moving_prefix_actor import statistics
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds


@pytest.mark.parametrize('prefix', [1, 3, 6, 12])
def test_registered_row_support_envelope_has_attainable_unknown_row_endpoints(prefix):
    old = np.zeros((3, 12, 2)); new = old.copy(); floor = old.copy()
    old[0, :, 0] = 3.; new[0, :, 0] = 1.; floor[0, :, 0] = 2.
    old[1, :, 0] = 5.; new[1, :, 0] = 2.; floor[1, :, 0] = 4.
    new[2, :, 0] = 4.
    target = np.full_like(old, np.nan); valid = np.zeros((3, 12), bool)
    valid[0] = True; valid[1, 0] = True; target[valid] = 0.
    weight = np.array([.25, .25, .5])
    row = next(r for r in statistics(old, new, floor, target, valid, weight, 5., 10.) if r['prefix'] == prefix)
    observed_reference = 1.5; observed_gain = 1.25; unknown_radius = 2.
    assert row['paired_known_pp'] == pytest.approx(100*observed_gain/observed_reference)
    assert row['paired_completion_lower_pp'] == pytest.approx(100*(observed_gain-unknown_radius)/observed_reference)
    assert row['paired_completion_upper_pp'] == pytest.approx(100*(observed_gain+unknown_radius)/observed_reference)
    assert row['rows'] == row['known_rows']+row['wholly_unknown_rows'] == 3
    assert row['partial_rows'] == (0 if prefix == 1 else 1)
    for endpoint, y in [('lower', old[2]), ('upper', new[2])]:
        unknown_gain = (np.linalg.norm(old[2, :prefix]-y[:prefix], axis=-1)
                        -np.linalg.norm(new[2, :prefix]-y[:prefix], axis=-1)).mean()
        completed = 100*(observed_gain+.5*unknown_gain)/observed_reference
        assert completed == pytest.approx(row[f'paired_completion_{endpoint}_pp'])


def test_partial_available_label_gain_does_not_certify_complete_grid_gain():
    old = np.zeros((1, 12, 2)); old[:, :, 0] = 3.
    new = old.copy(); new[:, :, 0] = 2.
    floor = old.copy(); floor[:, :, 0] = 1.
    target = np.full_like(old, np.nan); target[:, 0] = 0.
    valid = np.zeros((1, 12), bool); valid[:, 0] = True
    row = statistics(old, new, floor, target, valid, np.ones(1), 5., 10.)[1]
    assert row['prefix'] == 3 and row['partial_rows'] == 1
    assert row['paired_completion_lower_pp'] == 100.
    bounds = partial_gain_bounds(new[:, :3], old[:, :3], target[:, :3], valid[:, :3], np.ones(1))
    assert bounds['lower'][0] == pytest.approx(-1/3)
    assert bounds['upper'][0] == 1.
    completed = np.zeros((1, 3, 2)); completed[:, 1:, 0] = 100.
    gain = (np.linalg.norm(old[:, :3]-completed, axis=-1)
            -np.linalg.norm(new[:, :3]-completed, axis=-1)).mean()
    assert gain == pytest.approx(-1/3) and gain < 0


def test_full_grid_bounds_sharp_with_fixed_observations_and_causal_scale():
    rng = np.random.default_rng(174)
    old = rng.normal(size=(6, 12, 2)); new = rng.normal(size=old.shape)
    valid = rng.random((6, 12)) < .5; target = rng.normal(size=old.shape)
    scale = np.linspace(.5, 3., len(old))
    bounds = partial_gain_bounds(new, old, target, valid, scale)
    for key, fill in [('lower', old), ('upper', new)]:
        completed = np.where(valid[..., None], target, fill)
        gain = ((np.linalg.norm(old-completed, axis=-1)-np.linalg.norm(new-completed, axis=-1))*scale[:, None]).mean(1)
        np.testing.assert_allclose(gain, bounds[key], rtol=1e-13, atol=1e-13)


def test_empty_reference_stays_NA_even_for_identical_rollouts():
    z = np.zeros((2, 12, 2)); valid = np.zeros((2, 12), bool)
    for mask in (valid, ~valid):
        for row in statistics(z, z, z, z, mask, np.full(2, .5), 1., 2.):
            assert row['paired_known_pp'] is None
            assert row['paired_completion_lower_pp'] is None
            assert row['paired_completion_upper_pp'] is None
            assert row['treatment']['known_easy_positive_harm_percent'] is None


def test_paired_disagreement_is_not_sum_of_separate_floor_radii():
    baseline = np.zeros((1, 12, 2)); old = baseline.copy(); new = baseline.copy()
    old[..., 0] = 2.; new[..., 0] = 3.
    paired = np.linalg.norm(new-old, axis=-1).max()
    separate = np.linalg.norm(new-baseline, axis=-1).max()+np.linalg.norm(old-baseline, axis=-1).max()
    assert paired == 1. and separate == 5.


def test_selected_reference_ratio_need_not_decrease_with_fallback():
    reference = np.array([1., 100.]); harm = np.array([.03, 0.])
    assert harm.sum()/reference.sum() < .02
    assert harm[0]/reference[0] > .02
