import numpy as np
import pytest
from scripts.inspect_m3w_aux_initial_prior import constant_bce_excess


def test_constant_bce_excess_matches_cross_entropy_difference():
    p, q = .28, .05
    ce = -q*np.log(p)-(1-q)*np.log1p(-p)
    entropy = -q*np.log(q)-(1-q)*np.log1p(-q)
    assert constant_bce_excess(p,q) == pytest.approx(ce-entropy)
    assert constant_bce_excess(q,q) == 0


def test_degenerate_targets_valid_but_invalid_initialization_rejected():
    assert constant_bce_excess(.2,0) == pytest.approx(-np.log(.8))
    assert constant_bce_excess(.2,1) == pytest.approx(-np.log(.2))
    with pytest.raises(ValueError): constant_bce_excess(0,.2)
    with pytest.raises(ValueError): constant_bce_excess(.2,1.1)
