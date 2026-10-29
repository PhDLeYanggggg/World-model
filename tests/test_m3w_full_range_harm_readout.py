import copy
import numpy as np
import pytest

from src.evaluation import m3w_full_range_harm_readout as r
from tests.test_m3w_shared_train_risk import packet


def test_replay_and_missing_rows_not_hidden():
    args = list(packet())
    y = args[2].copy()
    y[0] = np.nan
    args[2] = y
    p = np.tile([1., .001, 1., 1., .001], (len(y), 1))
    value = r.evaluate(args, {a: p for a in r.ARMS}, np.ones(len(y), bool))
    r.validate(value)
    assert value['unknown_rows'] >= 1
    for a in r.CONTROLS:
        assert value['pairs'][a+'_full']['lower_mass'] == 0
    bad = copy.deepcopy(value)
    bad['pairs']['anchor_full']['lower_mass'] += 1
    with pytest.raises(ValueError):
        r.validate(bad)


def test_protected_reference_cannot_change():
    args = packet()
    p = np.tile([1., .001, 1., 1., .001], (len(args[0]), 1))
    ps = {a: p.copy() for a in r.ARMS}
    ps[r.TARGET][0, 2] = 100
    with pytest.raises(AssertionError):
        r.evaluate(args, ps, np.ones(len(p), bool))
