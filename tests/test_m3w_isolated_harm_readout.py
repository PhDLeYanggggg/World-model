import copy

import numpy as np
import pytest

from scripts import evaluate_m3w_isolated_harm as api
from tests.test_m3w_shared_train_risk import packet


def fixture():
    args = packet()
    p = np.tile([.3, .001, 1., 1., .001], (len(args[0]), 1))
    predictions = {a: p.copy() for a in api.ARMS}
    predictions['isolated_risk'][::2, 1] = .2
    predictions['isolated_risk'][::2, 4] = .2
    return args, predictions


def test_full_rows_fixed_cohort_and_unknown_actions():
    args, predictions = fixture()
    r = api.evaluate(args, predictions)
    api.check_result(r)
    assert r['unknown_rows'] == 1
    assert r['policies']['anchor']['selected_unknown'] == 1
    assert r['policies']['isolated_risk']['selected_count'] < r['policies']['anchor']['selected_count']
    assert r['fixed_anchor_cohort']['anchor']['observed_moments'] == r['fixed_anchor_cohort']['isolated_risk']['observed_moments']


def test_future_labels_do_not_change_inference_actions():
    args, p = fixture()
    a = api.evaluate(args, p)
    changed = list(copy.deepcopy(args))
    changed[2][:] = np.nan
    changed[3][:] = np.nan
    b = api.evaluate(changed, p)
    for arm in api.ARMS:
        assert a['policies'][arm]['selected_count'] == b['policies'][arm]['selected_count']


@pytest.mark.parametrize('corrupt', ['cohort', 'utility', 'counts', 'protection'])
def test_readout_corruption_rejected(corrupt):
    r = api.evaluate(*fixture())
    if corrupt == 'cohort':
        r['fixed_anchor_cohort']['isolated_risk']['known_selected'] += 1
    elif corrupt == 'utility':
        r['contrasts']['isolated_risk_minus_anchor_full'] += 1
    elif corrupt == 'counts':
        r['policies']['anchor']['selected_count'] += 1
    else:
        r['protected_outputs_exact'] = False
    with pytest.raises((ValueError, AssertionError)):
        api.check_result(r)


def test_protected_outputs_and_complete_roster_required():
    args, p = fixture()
    p['isolated_risk'][0, 0] += 1
    with pytest.raises(AssertionError):
        api.evaluate(args, p)
    del p['local']
    with pytest.raises(ValueError):
        api.evaluate(args, p)
