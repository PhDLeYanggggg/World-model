import numpy as np
import pytest
from src.evaluation import m3w_calibration_paired_effects as api


def test_actual_pair_distance_not_subtraction_of_individual_bounds():
    floor = np.zeros((1, 12, 2)); a = floor.copy(); b = floor.copy()
    a[..., 0] = -1; b[..., 0] = 1
    assert api.unknown_distance(a, b) == 2
    assert api.unknown_distance(b, floor)-api.unknown_distance(a, floor) == 0
    result = api.paired_values(1, 0, 0, api.unknown_distance(a, b))
    assert result == dict(known_pp=0, completion_lower_pp=-200, completion_upper_pp=200)


def test_matches_original_direct_paired_metric():
    from src.evaluation.m3w_selection_forecast_readout import paired, errors
    rng = np.random.default_rng(22)
    y = rng.normal(size=(5, 12, 2)); a = y+.1; b = y+.3; floor = y+.8
    valid = np.ones((5, 12), bool); valid[1, 3:8] = False; valid[-1] = False
    aa, _ = errors(a, y, valid); bb, _ = errors(b, y, valid); rr, _ = errors(floor, y, valid)
    known = valid.any(1)
    result = api.paired_values(float(rr[known].sum()), float((rr[known]-aa[known]).sum()),
        float((rr[known]-bb[known]).sum()), api.unknown_distance(a[~known], b[~known]))
    direct = paired(a, b, floor, y, valid)
    np.testing.assert_allclose([result[k] for k in result],
                              [direct[k] for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')])


def test_zero_support_not_safe_zero_and_bad_inputs_rejected():
    assert api.paired_values(0, 0, 0, 0)['known_pp'] is None
    with pytest.raises(ValueError):
        api.paired_values(1, 0, 0, -1)
    with pytest.raises(ValueError):
        api.unknown_distance(np.zeros((1, 2)), np.zeros((1, 2)))
    assert api.interval(dict(a=None))['point'] is None
