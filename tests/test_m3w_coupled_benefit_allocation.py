import copy
import numpy as np
import pytest
from src.world_model import m3w_coupled_benefit_allocation as api
from src.world_model import m3w_critic_allocation as old
from src.evaluation import m3w_coupled_benefit_risk as evaluation
from tests.test_m3w_critic_allocation import inputs


def fixture():
    p, allowed, rec, frames, ids = inputs()
    control = old.allocate(dict(row_local=p['row_local'], selected_set=p['row_local']), allowed, rec, frames, ids, seconds=2.)[0]
    changed = p['selected_set'].copy(); changed[:, :, 0] *= .7
    return dict(protected_benefit=p['row_local'], released_benefit=changed), allowed, rec, frames, ids, control


def test_different_benefits_shared_references_and_real_count_controls():
    args = fixture(); choices, ranks, info = api.allocate(*args, seconds=2.)
    np.testing.assert_array_equal(choices['protected_joint'], args[-1]['set_joint'])
    assert not info['labels_used'] and info['same_count_controls_every_query']
    assert info['protected_full_choices'] == 'cached_verified_not_reoptimized'
    assert set(choices) == set(ranks) == set(api.NAMES)
    for g in api.base.query_groups(*args[2:5]):
        for prefix, mode in zip(('protected', 'released'), api.MODES):
            for suffix in ('independent', 'joint', 'matched'):
                assert api.base.feasible(args[0][mode][g], args[1][g], choices[prefix+'_'+suffix][g])
        assert (choices['protected_matched'][g] >= 0).sum() == (choices['released_matched'][g] >= 0).sum()
    bad = fixture(); bad[0]['released_benefit'][0, 0, 2] += .01
    with pytest.raises(AssertionError): api.allocate(*bad, seconds=2.)


def test_resume_exact_and_changed_control_refuses():
    args = fixture(); saved = []
    class Stop(Exception): pass
    def checkpoint(value): saved.append(copy.deepcopy(value)); raise Stop()
    with pytest.raises(Stop): api.allocate(*args, seconds=2., checkpoint=checkpoint)
    direct = api.allocate(*args, seconds=2.)
    resumed = api.allocate(*args, seconds=2., resume_state=saved[-1])
    old.exact(direct, resumed)
    bad = copy.deepcopy(args); bad[-1]['set_joint'][0] = -1
    with pytest.raises((ValueError, AssertionError)): api.allocate(*bad, seconds=2., resume_state=saved[-1])


def test_summary_refuses_small_or_missing_sources():
    with pytest.raises(ValueError, match='All72'):
        evaluation.summarize([], {}, dict(seeds=[17, 29, 43]))
