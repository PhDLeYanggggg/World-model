import copy

import numpy as np
import pytest

from src.evaluation import m3w_query_context_readout as api
from tests.test_m3w_shared_train_risk import packet


def test_identical_policies_zero_pair_and_unknown_preserved():
    args = packet(); n = len(args[0])
    pred = np.tile([.5, 0., 1., 1., 0.], (n, 1))
    pp = {a: pred.copy() for a in api.ARMS}
    r = api.evaluate(args, pp, np.ones(n, bool)); api.validate(r)
    assert all(r['contrasts'][a+'_matched'] == 0 for a in api.CONTROLS)
    assert r['policies']['peer']['selected_unknown'] == 1
    altered = copy.deepcopy(r); altered['pairs']['self_full']['lower_mass'] += 1
    with pytest.raises(ValueError): api.validate(altered)


def test_unknown_difference_is_penalized_and_unregistered_arm_rejected():
    args = packet(); n = len(args[0])
    pred = np.tile([.5, 0., 1., 1., 0.], (n, 1))
    pp = {a: pred.copy() for a in api.ARMS}
    pp['peer'][-1, 1] = 1
    r = api.evaluate(args, pp, np.ones(n, bool)); api.validate(r)
    assert r['pairs']['self_full']['known_difference_mass'] == 0
    assert r['contrasts']['self_full'] < 0
    del pp['local']
    with pytest.raises(ValueError): api.evaluate(args, pp, np.ones(n, bool))
