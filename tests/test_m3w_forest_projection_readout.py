import copy

import numpy as np
import pytest

from scripts.verify_m3w_forest_projection import check_group, close
from src.world_model import m3w_forest_projection as api
from src.world_model.m3w_source_forest import project_moments


def fixture():
    raw = np.array([[20., 1., 40., 40., 1.], [10., .01, 20., 20., .01]])
    env = np.array([2., 12.])
    y = np.array([[0., 1., 40., 40., 1.], [np.nan]*5])
    b = np.ones(2, bool)
    result, _, _ = api.audit(raw, project_moments(raw, env), api.harm_first(raw, env),
        y, env, b, b, np.array(['a','b']), np.array([0,0]))
    return dict(result=result, hashes=dict(targets='synthetic'), inputs_source='cached_verified')


def test_independent_arithmetic_on_known_and_unknown():
    assert check_group(fixture()) == 44


def test_forged_mass_is_rejected():
    g = copy.deepcopy(fixture())
    g['result']['removed']['selected_known_harm_mass'] = 0
    with pytest.raises(AssertionError):
        check_group(g)


def test_undefined_is_not_zero():
    with pytest.raises(AssertionError):
        close(None, 0.)
