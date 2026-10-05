import copy
import inspect
import itertools
import numpy as np
import pytest

from src.world_model import m3w_temporal_auxiliary_readout as api
from scripts import verify_m3w_temporal_auxiliary_readout as scalar


def fixture():
    y = np.array([[2, 0, 10, 10, 0], [0, 1, 10, 10, 1],
                  [0, 1, 0, 0, 1], [np.nan]*5, [1, 0, 4, 0, 0], [0, 0, 5, 5, 0.]])
    env = np.full(6, 3.)
    p = np.tile([1., .01, 10., 10., .01], (6, 1))
    pred = {k: p.copy() for k in api.ARMS}
    pred['temporal'][[1, 2], 1] = 1.
    pred['temporal'][4, 0] = 2.
    pred['rowmean'][0, 0] = 0.
    pr = dict(scale=2., rms=np.arange(1., 9))
    return pr, pred, y, env, np.ones(6, bool), np.ones(6, bool), np.array(['a']*6), np.array([3, 3, 3, 4, 4, 4]), np.array([1, 1, 2, 1, 1, 2]), np.array([9, 1, 8, 2, 7, 6])


def test_full_and_matched_readout_independent_scalar_verification():
    args = fixture()
    result, actions = api.evaluate(*args)
    assert scalar.verify(*args, result, actions) > 500
    assert result['policies']['temporal']['selected_unknown'] == 1
    assert result['policies']['original']['selected_known_easy_harm_mass'] == 2
    assert result['policies']['temporal']['known_easy_positive_risk'] == 0
    assert result['policies']['temporal']['easy_selected_risk_upper'] > .02
    assert not result['policies']['temporal']['finite_completion_supported']


def test_actions_do_not_depend_on_outcome_availability_or_future_cost():
    args = list(fixture()); a = api.evaluate(*args)[1]
    args[2] = np.full_like(args[2], np.nan)
    result, b = api.evaluate(*args)
    for name in a: np.testing.assert_array_equal(a[name], b[name])
    assert result['contrasts']['original_all_MSE'] is None
    assert result['policies']['temporal']['easy_selected_risk_upper'] is None
    assert 'targets' not in inspect.signature(api.decisions).parameters


@pytest.mark.parametrize('missing', api.ARMS)
def test_missing_strong_or_trained_control_fails_closed(missing):
    args = list(fixture()); del args[1][missing]
    with pytest.raises(ValueError): api.evaluate(*args)


def test_same_query_matching_and_id_tie_break_not_global_matching():
    args = list(fixture())
    args[1]['temporal'][:, 1] = 1.
    args[1]['temporal'][[0, 4], 1] = .01
    result, actions = api.evaluate(*args)
    assert set(np.flatnonzero(actions['original_matched_temporal'])) == {1, 3}
    assert set(np.flatnonzero(actions['temporal_matched_original'])) == {0, 4}
    assert scalar.verify(*args, result, actions) > 0


def test_paired_unknown_completion_is_not_difference_of_lower_bounds():
    y = np.full((3, 5), np.nan); env = np.array([2., 3., 4.])
    a, b = np.array([True, True, False]), np.array([True, False, True])
    pair = api.paired_completion(y, a, b, env)
    assert pair['lower_mass'] == -7 and pair['upper_mass'] == 7
    assert pair['shared_unknown_selected'] == 1
    assert api.paired_completion(y, a, a, env)['lower_mass'] == 0
    attainable = [sum((int(bb)-int(aa))*v*e for aa, bb, v, e in zip(a, b, signs, env))
                  for signs in itertools.product((-1, 1), repeat=3)]
    assert min(attainable) == pair['lower_mass'] and max(attainable) == pair['upper_mass']
    old_proxy = api.completion_bounds(y, b, env)['selected_net_gain_lower_mass']-api.completion_bounds(y, a, env)['selected_net_gain_lower_mass']
    assert old_proxy == -1 and old_proxy != pair['lower_mass']


def test_undefined_reference_and_empty_original_cohort_do_not_get_zero_error():
    args = list(fixture())
    args[1]['original'][:, 0] = 0.
    r, _ = api.evaluate(*args)
    assert r['scores']['temporal']['original_selected']['conditional_MSE'] is None
    assert r['contrasts']['original_original_selected_MSE'] is None
    assert r['policies']['temporal_matched_original']['easy_selected_risk_upper'] is None


def summary_fixture():
    r, _ = api.evaluate(*fixture())
    rows = [dict(group=f'context{c}_site{s}', source=f'site{s}', head_seed=seed,
                 result=copy.deepcopy(r)) for s in range(12) for c in range(2) for seed in (17, 29, 43)]
    expected = [{k: row[k] for k in ('group', 'source', 'head_seed')} for row in rows]
    cfg = dict(source_heads=72, neural_fits=216, head_seeds=[17, 29, 43], bootstrap_resamples=3000,
               bootstrap_seed=20261005, risk_budget=.02)
    return rows, expected, cfg


def test_favorable_losses_do_not_override_risk_failure():
    rows, expected, cfg = summary_fixture()
    for r in rows:
        r['result']['contrasts'] = {k: -1. if k.endswith('_MSE') else 1. for k in r['result']['contrasts']}
    s = api.summarize(rows, expected, cfg)
    assert not s['advance_to_transfer_design']
    assert any('absolute_original_risk' in f for f in s['failure_reasons'])
    assert s['deployment_changed'] is False
    assert s['safety']['temporal']['easy_upper_violations'] == 72


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_seed'])
def test_no_partial_grid_claim(mutation):
    rows, expected, cfg = summary_fixture()
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows[-1] = rows[0]
    if mutation == 'wrong_seed': rows[0]['head_seed'] = 1
    with pytest.raises(ValueError): api.summarize(rows, expected, cfg)


def test_interval_resamples_twelve_localities_not_rows_or_seventy_two_heads():
    rows, _, cfg = summary_fixture()
    for row in rows: row['result']['contrasts']['test'] = int(row['source'][4:])+row['head_seed']/100
    out = api.interval(rows, 'test', cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    means = [np.mean([r['result']['contrasts']['test'] for r in rows if r['source'] == s])
             for s in sorted({r['source'] for r in rows})]
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    boot = [sum(means[i] for i in rng.integers(0, 12, 12))/12 for _ in range(3000)]
    np.testing.assert_allclose(out['CI95'], np.quantile(boot, [.025, .975]), rtol=1e-12)
    assert len(out['localities']) == 12
    rows[0]['result']['contrasts']['test'] = None
    assert api.interval(rows, 'test', 3000, cfg['bootstrap_seed'])['CI95'] is None


def test_scalar_verifier_detects_wrong_risk_or_prediction_scoring():
    args = fixture(); result, actions = api.evaluate(*args)
    result['policies']['temporal']['easy_selected_risk_upper'] = 0.
    with pytest.raises(AssertionError): scalar.verify(*args, result, actions)
