import numpy as np
import pytest

from src.evaluation.m3w_frozen_amplitude import bounds, shrink, total_risk_cause, trust_alpha
from src.evaluation.m3w_selection_forecast_readout import policy


def test_convex_bounds_keep_selected_and_unknown_reference_contract():
    floor = np.ones((3, 2, 2))
    target = np.zeros_like(floor)
    alt = floor.copy()
    alt[0] = -floor[0]     # Equal endpoint errors; the midpoint improves ADE.
    alt[1] = 4.*floor[1]  # Wholly unknown row must still pay disagreement.
    valid = np.array([[True, True], [False, False], [True, False]])
    rec = np.array(['r']*3)
    risk = policy(shrink(floor, alt, .5), floor, floor, target, valid, rec,
                  easy_cut=3., hard_cut=4.)['recordings']['r']
    assert risk['selected'] == 2 and risk['unknown_selected'] == 1
    bound = bounds(floor, alt, target, valid, 3., .5)
    for k, v in bound.items():
        assert risk[k] <= v
    assert total_risk_cause(risk) == 'missing_radius_crosses_2'


def test_past_scale_caps_disagreement_without_zeroing_switches():
    floor = np.zeros((2, 3, 2))
    alt = floor.copy()
    alt[0, -1, 0] = 6.
    alpha = trust_alpha(floor, alt, np.array([2., 3.]))
    np.testing.assert_allclose(alpha, [1./3., 1.])
    np.testing.assert_allclose(shrink(floor, alt, alpha)[0, -1], [2., 0.])
    with pytest.raises(ValueError, match='strictly positive'):
        shrink(floor, alt, 0.)


def test_absent_easy_denominator_is_not_restored_by_shrink():
    floor = np.zeros((1, 2, 2))
    valid = np.zeros((1, 2), bool)
    assert all(v is None for v in bounds(floor, floor+1., floor, valid, 2., .25).values())
