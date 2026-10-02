import copy
import numpy as np
import pytest
from scripts.verify_m3w_quality_components import verify_cohorts
from src.world_model import m3w_quality_component_diagnostic as api


def result():
    raw = np.tile([.5, .01, 3., 3., .01], (4, 1))
    p = api.variants(raw, np.zeros_like(raw), np.ones(4))
    y = np.tile([.2, 0., 2., 2., 0.], (4, 1)); y[-1] = np.nan
    return dict(result=api.evaluate(p, y, np.ones(4), np.ones(4, bool), np.ones(4, bool),
                                   np.array([0, 0, 1, 1]), np.zeros(4), np.arange(4)))


def test_scalar_cohort_identity_and_completion_verification():
    assert verify_cohorts(result()) > 500


def test_false_support_pass_is_rejected():
    r = copy.deepcopy(result())
    assert not r['result']['arms']['quality']['full']['finite_completion_supported']
    r['result']['arms']['quality']['full']['finite_completion_supported'] = True
    with pytest.raises(AssertionError): verify_cohorts(r)
