import math
import json

import numpy as np
import pytest

from src.world_model.m3w_net_easy_guard import allocate, validate_receipt, require_replay_outputs


def test_mixed_magnitude_budget_is_not_relaxed():
    b, r = allocate([1., 2.], [-1000., 1e-6], np.ones(2, bool), 1e-10, count=1)
    np.testing.assert_array_equal(b, [True, False])
    assert r['optimal'] and r['constraint_pass'] and r['exact_count_pass']
    assert math.fsum(np.array([-1000., 1e-6])[b]) <= 1e-10
    assert json.loads(json.dumps(r))['exact_count_pass'] is True


def test_larger_invalid_solution_fails_closed():
    b, r = allocate(np.r_[1., np.full(18, 2.)], np.r_[-1000., np.full(18, 1e-6)],
                    np.ones(19, bool), 1e-10, count=1)
    assert not b.any() and not r['optimal'] and not r['exact_count_pass']


def fixture():
    identity = dict(view='coupa_seed17', action='transformer', experiment_sha256='exp', frozen_head_sha256='old')
    cp = dict(identity=identity, seed=17, preprocess=dict(training_sites=['gates']))
    return dict(identity=identity), cp


def test_receipt_correct_identity():
    r, cp = fixture()
    validate_receipt(r, 'coupa_seed17', 'transformer', 'exp', dict(checkpoint_sha256='old'), cp)


@pytest.mark.parametrize('view,action,old', [('coupa_seed29','transformer','old'),
    ('coupa_seed17','eqmotion','old'), ('coupa_seed17','transformer','wrong')])
def test_receipt_wrong_identity_rejected(view, action, old):
    r, cp = fixture()
    with pytest.raises(ValueError):
        validate_receipt(r, view, action, 'exp', dict(checkpoint_sha256=old), cp)


def test_first_evaluation_cannot_be_replay(tmp_path):
    with pytest.raises(ValueError):
        require_replay_outputs(tmp_path, tmp_path, ['transformer'], [17])
