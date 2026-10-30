import copy
import numpy as np
import pytest
from src.world_model import m3w_critic_allocation as api


def inputs():
    rng = np.random.default_rng(5)
    prediction = {}
    for arm in ('row_local', 'selected_set'):
        p = np.empty((9, 7, 5))
        p[:, :, 0] = .2; p[:, :, 2] = 1.; p[:, :, 3] = .5
        p[:, :, 1] = rng.uniform(0, .04, (9, 7))
        p[:, :, 4] = p[:, :, 1]*rng.uniform(0, 1, (9, 7))
        prediction[arm] = p
    return prediction, np.ones((9, 7), bool), np.repeat('r', 9), np.repeat([0, 1, 2], 3), np.arange(9)


def test_common_counts_and_feasible_raw_budgets():
    args = inputs(); choices, ranks, info = api.allocate(*args, seconds=2.)
    assert info['same_count_controls_every_query'] and not info['labels_used']
    for g in api.base.query_groups(*args[2:]):
        assert (choices['row_matched'][g] >= 0).sum() == (choices['set_matched'][g] >= 0).sum()
        for prefix, arm in (('row', 'row_local'), ('set', 'selected_set')):
            for suffix in ('independent', 'joint', 'matched'):
                assert api.base.feasible(args[0][arm][g], args[1][g], choices[prefix+'_'+suffix][g])
    assert set(choices) == set(ranks) == set(api.NAMES)


def test_interrupted_recovery_exact_and_changed_prediction_rejected():
    args = inputs(); saved = []
    class Stop(Exception): pass
    def stop(value):
        saved.append(copy.deepcopy(value)); raise Stop()
    with pytest.raises(Stop): api.allocate(*args, seconds=2., checkpoint=stop)
    direct = api.allocate(*args, seconds=2.)
    resumed = api.allocate(*args, seconds=2., resume_state=saved[-1])
    api.exact(direct, resumed)
    changed = inputs(); changed[0]['row_local'][0, 0, 1] += .001
    with pytest.raises(ValueError):
        api.allocate(*changed, seconds=2., resume_state=saved[-1])


def test_benefit_and_references_must_be_shared():
    args = inputs(); args[0]['selected_set'][0, 0, 2] += .001
    with pytest.raises(AssertionError): api.allocate(*args, seconds=2.)
