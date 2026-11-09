import numpy as np
import pytest

from scripts.audit_m3w_record_peak_break_even import minimum_peak
from src.evaluation.m3w_record_peak_allocation import radius_envelope


def test_minimum_peak_inverts_fractional_envelope():
    cap = np.array([[0., 1., 8.], [1., 2., 4.], [2., 2., 4.]])
    w = np.array([.5, .5, 1.])
    for gain in (0., .25, 2., 9.3, 16.):
        peak = minimum_peak(cap, w, gain)
        assert radius_envelope(cap, w, peak) == pytest.approx(gain)
        if peak > 0.:
            assert radius_envelope(cap, w, peak - .001) < gain
    assert minimum_peak(cap, w, 20.) is None
