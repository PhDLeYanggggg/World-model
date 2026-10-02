import inspect
import itertools

import numpy as np
import pytest

from src.world_model import m3w_recording_stability as api
from tests.test_m3w_selected_set_readout import fixture


@pytest.mark.parametrize('arm', api.ARMS)
def test_nested_coefficients_and_actions_exclude_held_and_deleted_outcomes(arm):
    z, _, _, cfg = fixture()
    a = api.fit_excluding(z, ['a', 'b'], arm, cfg)
    z['y'][z['recordings'] != 'c'] = [0., .9, .01, .01, .9]
    b = api.fit_excluding(z, ['a', 'b'], arm, cfg)
    assert a == b and set(a['recordings']) == {'c'}
    assert set(inspect.signature(api.infer_consensus).parameters) == {
        'p', 'env', 'moving', 'support', 'base', 'deletions'}


def test_realistic_fixture_exact_replay_and_existing_parent_anchor():
    z, g, old, cfg = fixture()
    result = api.audit_head(z, old, g, cfg)
    assert result == api.audit_head(z, old, g, cfg)
    for arm, r in result['arms'].items():
        assert r['parent_hash'] == r['consensus_hash']
        assert r['parent']['selected_count'] == 9
        assert r['unique_calibrators'] == 6
        assert len(r['folds']) == 3 and all(f['estimable'] for f in r['folds'])


def test_unestimable_deletion_is_not_empty_success():
    z, _, _, cfg = fixture()
    base = api.fit_excluding(z, ['a'], 'raw_pool', cfg)
    empty = api.fit_excluding(z, ['a', 'b', 'c'], 'raw_pool', cfg)
    b, a, _, estimable = api.infer_consensus(z['p'], z['env'], z['moving'], z['support'], base, [empty])
    assert b.all() and not a.any() and not estimable
    bounds = api.check.scalar_bounds(z['y'], a, z['env'])
    assert bounds['easy_selected_risk_upper'] is None
    assert not bounds['finite_completion_supported']


def test_supported_conservative_nested_action_cannot_add_rows():
    z, _, _, cfg = fixture()
    base = api.fit_excluding(z, ['a'], 'raw_pool', cfg)
    reject = dict(base, margins=[.9, .9, 0., 0.])
    b, a, choices, estimable = api.infer_consensus(z['p'], z['env'], z['moving'], z['support'], base, [base, reject])
    assert estimable and b.all() and not a.any() and len(choices) == 2


def test_unknown_evaluation_labels_do_not_change_fixed_inference():
    z, _, _, cfg = fixture()
    cal = api.fit_excluding(z, ['a'], 'raw_pool', cfg)
    args = [z[k] for k in ('p', 'env', 'moving', 'support')]+[cal, [cal]]
    before = api.infer_consensus(*args)
    z['y'][:] = np.nan
    after = api.infer_consensus(*args)
    np.testing.assert_array_equal(before[1], after[1])
    bounds = api.check.scalar_bounds(z['y'], after[1], z['env'])
    assert bounds['selected_unknown'] == 9 and not bounds['finite_completion_supported']


def test_count_matched_utility_expectation_equals_exhaustive_subsets():
    y = np.array([[1., 0., 1., 1., 0.], [0., .4, 1., 1., .4], [np.nan]*5, [2., 0., 2., 2., 0.]])
    env = np.array([1., 1., 3., 2.])
    orig = np.ones(4, bool)
    retain = np.array([True, False, False, True])
    got = api.matched_expectation(y, env, orig, retain, np.array(['a']*3+['b']))
    vals = []
    for chosen in itertools.combinations(range(3), 1):
        a = np.zeros(4, bool); a[list(chosen)+[3]] = True
        vals.append(api.check.scalar_bounds(y, a, env)['selected_net_gain_lower_mass'])
    assert got['conservative_utility_expectation'] == pytest.approx(np.mean(vals))
    assert got['selected_occurrences'] == 2 and got['ratio_or_safety_guarantee'] is False


def test_matched_expectation_rejects_non_subset():
    z, _, _, _ = fixture()
    with pytest.raises(ValueError):
        api.matched_expectation(z['y'], z['env'], np.zeros(9, bool), np.ones(9, bool), z['recordings'])


def test_frozen_oof_tampering_rejected():
    z, g, old, cfg = fixture()
    old['folds'][0]['calibration']['margins'][0] = .8
    with pytest.raises(AssertionError):
        api.audit_head(z, old, g, cfg)


def test_locality_bootstrap_averages_dependent_heads_first():
    cfg = dict(bootstrap_resamples=3000, bootstrap_seed=20261002)
    v = api.interval([('a', 1.), ('a', 3.), ('b', 10.), ('c', None)], cfg)
    assert v['mean'] == 6. and v['localities'] == 2
    assert v['undefined_views'] == 1 and v['strict_all_views_mean'] is None
    assert v == api.interval([('a', 1.), ('a', 3.), ('b', 10.), ('c', None)], cfg)


def test_summary_empty_consensus_preserves_unknown_and_no_safety_claim():
    z, g, old, cfg = fixture()
    result = api.audit_head(z, old, g, cfg)
    report = api.summarize([result], cfg)
    assert report['groups'] == 1 and not report['deployment_promoted']
    assert report['arms']['raw_pool']['utility_change_vs_parent']['mean'] == 0.
    assert report['arms']['raw_pool']['utility_change_vs_matched_expectation']['mean'] == 0.
