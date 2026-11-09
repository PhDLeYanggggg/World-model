import numpy as np

from scripts.diagnose_m3w_vehicle_pair_support import witness


def test_pair_hindsight_requires_support_distinct_recipients_and_full_ols_gain():
    rows = np.array([0, 0, 1, 2])
    risk = np.zeros((4, 3))
    ref = np.array([1., 1., 0., 0.])
    gain = np.array([[3., 1.], [3., 1.], [2., 0.], [.2, 4.]])
    np.testing.assert_array_equal(witness(rows, risk, ref, gain, 0., -2.), [0, 3])
    assert witness(rows, risk, ref, gain, 0., -5.) is None
    assert witness(rows, risk, ref*0., gain, 0., 0.) is None
