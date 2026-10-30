import copy
import numpy as np
import pytest
from src.evaluation import m3w_active_bank_cost_readout as api


def fixture(source='s0', missing=False):
    n = 2
    known = np.array([True, not missing])
    paths = np.ones((n, 7, 12, 2))
    paths[..., 1] = 0.
    floor = paths[:, 0].copy()
    paths[:, 6, :, 0] = .8
    ade = np.ones((n, 7))
    ade[:, 6] = .8
    ade[~known] = np.nan
    y = np.zeros((n, 7, 5))
    y[..., 2:4] = 1.
    y[:, 6, 0] = .2
    y[~known] = np.nan
    envelope = np.linalg.norm(paths - floor[:, None], axis=-1).max(2)
    allowed = np.zeros((n, 7), bool)
    allowed[:, 6] = True
    old = dict(ids=np.arange(n), sites=np.array([source] * n), recordings=np.array(['r0'] * n),
        frames=np.arange(n), scales=np.ones(n), floor_rollout=floor, floor_ade=ade[:, 0].copy(),
        floor_fde=ade[:, 0].copy(), cv_ade=ade[:, 0].copy(), hard=np.zeros(n, bool),
        rollouts=paths, ade=ade, fde=ade.copy(), y=y, envelope=envelope, allowed=allowed)
    new = copy.deepcopy(old)
    new['rollouts'][:, 6, :, 0] = .4
    new['ade'][known, 6] = .4
    new['fde'][known, 6] = .4
    new['y'][known, 6, 0] = .6
    new['envelope'][:, 6] = .6
    packets = dict(physical_only=old, old_neural=old, new_neural=new)
    choices = dict(physical_only=np.full(n, -1), old_neural=np.full(n, 6), new_neural=np.full(n, 6))
    ranks = {v: np.ones(n) for v in api.VARIANTS}
    predictions = {v: np.nan_to_num(packets[v]['y']) for v in api.VARIANTS}
    return packets, choices, ranks, predictions


def test_same_neural_action_index_must_compare_different_paths_and_unknown_bounds():
    packets, choices, _, _ = fixture(missing=True)
    old, new = packets['old_neural'], packets['new_neural']
    wrong = api.legacy.paired_bounds(old, choices['old_neural'], choices['new_neural'])
    assert wrong['known_pp'] == 0.
    proper = api.cross_bank_bounds(old, choices['old_neural'], new, choices['new_neural'])
    assert proper['known_pp'] == pytest.approx(40.)
    assert proper['lower_pp'] == pytest.approx(0.)
    assert proper['upper_pp'] == pytest.approx(80.)
    new['ids'][0] += 20
    with pytest.raises(AssertionError):
        api.cross_bank_bounds(old, choices['old_neural'], new, choices['new_neural'])


def test_empty_count_match_is_not_estimable_and_old_policy_core_is_exact():
    packets, choices, ranks, predictions = fixture()
    old = packets['old_neural']
    baseline_choices = dict(physical_only=choices['physical_only'], physical_neural=choices['old_neural'],
        neural_only=choices['old_neural'], frozen_origin_binary=choices['old_neural'])
    old_result = api.legacy.group_readout(old, baseline_choices,
        {v: np.ones(2) for v in baseline_choices}, {})
    result = api.group_readout(packets, choices, ranks, predictions, 1.5)
    assert result['policies']['old_neural'] == old_result['policies']['physical_neural']
    assert result['policies']['physical_only'] == old_result['policies']['physical_only']
    empty = result['comparisons']['new_minus_physical']
    assert empty['matched_empty'] and empty['matched']['known_pp'] is None
    assert empty['matched']['reason'] == 'not_estimable_empty_count_match'
    assert not result['policies']['physical_only']['policy']['budget_status']['supported']
    assert result['comparisons']['new_minus_old']['full']['known_pp'] == pytest.approx(40.)


def records(missing=False):
    contexts = {f'c{i}': [f's{4 * (i % 3) + j}' for j in range(4)] for i in range(6)}
    rows = []
    for context, sources in contexts.items():
        for source in sources:
            result = api.group_readout(*fixture(source, missing), 1.5)
            for seed in (17, 29, 43):
                rows.append(dict(context=context, source=source, seed=seed,
                    legacy_readout_exact=True, result=copy.deepcopy(result)))
    return rows, contexts, dict(seeds=[17, 29, 43], bootstrap_seed=2, bootstrap_resamples=32)


def test_full72groups_and_passing_development_gate_do_not_promote_deployment():
    rows, contexts, cfg = records()
    result = api.summarize(rows, contexts, cfg, dict(count=54, legacy_exact_endpoints=36))
    assert result['fitting_advancement_gate']
    assert result['accuracy']['new_neural']['point'] == pytest.approx(60.)
    assert not result['independent_confirmation'] and not result['deployment_changed']
    assert not result['stage5c_executed'] and not result['smc_enabled']
    assert result['bank_opportunity_diagnostic']['new_neural']['positive_record_views'] == 24
    assert result['subsets']['hard']['macro_status'] == 'not_estimable'
    with pytest.raises(ValueError):
        api.summarize(rows[:-1], contexts, cfg, dict(count=54, legacy_exact_endpoints=36))
    with pytest.raises(ValueError):
        api.summarize(rows[:-1] + rows[:1], contexts, cfg, dict(count=54, legacy_exact_endpoints=36))


def test_positive_known_utility_cannot_hide_unknown_risk_or_missing_fde():
    rows, contexts, cfg = records(missing=True)
    for row in rows:
        if row['source'] == 's0':
            row['result']['policies']['new_neural']['known_FDE_gain_floor_percent'] = None
    result = api.summarize(rows, contexts, cfg, dict(count=54, legacy_exact_endpoints=36))
    assert result['comparisons']['new_minus_old']['full_known_pp']['point'] > 0
    assert not result['fitting_advancement_gate']
    assert result['safety']['new_neural']['supported_record_views'] == 0
    assert result['FDE']['new_neural']['point'] is None
