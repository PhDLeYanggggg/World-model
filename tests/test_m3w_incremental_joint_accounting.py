import numpy as np
import pytest
from scripts.diagnose_m3w_european_incremental_joint import accounting
from scripts.verify_m3w_european_incremental_joint import reference_costs


def test_fewer_switches_can_increase_error():
    r = accounting([0., 1.], [3., 1.], [1., 1.], np.array([True, True]), np.array([False, True]))
    assert r['lost_benefit_percent_CV'] == 150 and r['avoided_harm_percent_CV'] == 0
    assert r['half_degradation_percent_CV'] > r['full_degradation_percent_CV']


@pytest.mark.parametrize('seed', range(12))
def test_accounting_identity(seed):
    rng = np.random.default_rng(seed); n, f, cv = rng.random((3, 100))+.01
    full = rng.random(100) < .6; half = full & (rng.random(100) < .5)
    r = accounting(n, f, cv, full, half)
    np.testing.assert_allclose(r['count_thinning_error_change_percent_CV'],
        r['lost_benefit_percent_CV']-r['avoided_harm_percent_CV'])


def test_unknown_is_not_zero_and_additions_not_thinning():
    with pytest.raises(ValueError): accounting([np.nan], [1.], [1.], np.array([True]), np.array([False]))
    with pytest.raises(ValueError): accounting([1.], [1.], [1.], np.array([False]), np.array([True]))


def test_verifier_normalizes_in_production_float64_precision():
    u = np.array([[.1234567, .1000001], [.7654321, .1234567]], np.float32)
    risk = u/7; scale = 123.4567
    gain, harm = reference_costs(u, risk, scale)
    assert gain.dtype == harm.dtype == np.float64
    np.testing.assert_array_equal(gain, np.array([(float(a)-float(b))/scale for a, b in u]))
    np.testing.assert_array_equal(harm, np.array([float(a)/scale for a in risk[:, 1]]))
