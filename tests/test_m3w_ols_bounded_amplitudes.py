import numpy as np

from scripts.report_m3w_ols_bounded_amplitudes import match_amplitudes


def test_exact_amplitude_matching_resolves_two_actions_for_same_recipient():
    s = np.array([[10.,8.],[9.,1.],[2.,1.5]])
    q = [np.arange(3)]
    r = match_amplitudes(s,np.ones_like(s,dtype=bool),q,np.array([-1,1,0]),2)
    np.testing.assert_array_equal(r,[1,0,-1])


def test_no_reference_switches_never_activates():
    s = np.ones((2,2))
    np.testing.assert_array_equal(match_amplitudes(s,s>0,[np.arange(2)],[-1,-1],2),[-1,-1])
