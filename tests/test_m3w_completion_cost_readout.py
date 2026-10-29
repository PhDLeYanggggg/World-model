import copy
import numpy as np
import pytest

from src.evaluation import m3w_completion_cost_readout as api


def population():
    contexts = {f'c{i}': [dict(packet=dict(identity=dict(source=f's{j}')))
        for j in range((i % 3)*4, (i % 3)*4+4)] for i in range(6)}
    cfg = dict(seeds=[17, 29, 43], bootstrap_seed=11, bootstrap_resamples=32)
    y = np.broadcast_to([.1, 0., 1., 1., 0.], (3, 7, 5)).copy(); y[-1] = np.nan
    env = np.ones((3, 7)); score = api.base.compact_score(y, env, np.array([0, -1, -1]))
    rows = []
    for context, source, seed, mode in sorted(api.expected_roster(contexts, cfg['seeds'])):
        names = api.model.BANKS
        policies = {name: dict(policy=copy.deepcopy(score), known_ADE_gain_floor_percent=1.,
                              known_FDE_gain_floor_percent=1.) for name in names}
        bounds = {m: dict(known_pp=1., lower_pp=.5, upper_pp=1.5) for m in ('full', 'matched')}
        subset = dict(reference_ADE_mass=1., reference_FDE_mass=1., known_rows=2, known_FDE_rows=2,
            policies={n: dict(ADE_error_mass=.9, FDE_error_mass=.9, switched=1) for n in names})
        result = dict(policies=policies, recordings={'record': {n: copy.deepcopy(score) for n in names}},
            parent_contrasts={n: copy.deepcopy(bounds) for n in names},
            supervision_contrasts={n: copy.deepcopy(bounds) for n in names},
            comparisons={'physical_neural_minus_physical_only': copy.deepcopy(bounds)},
            subsets={s: copy.deepcopy(subset) for s in ('hard', 'easy')})
        rows.append(dict(context=context, source=source, seed=seed, mode=mode, result=result))
    return rows, contexts, cfg, y, env


def test_even_positive_fitting_gates_cannot_promote_deployment():
    rows, contexts, cfg, _, _ = population()
    result = api.summarize(rows, contexts, cfg)
    assert result['fitting_advancement_gate']
    assert not result['submission_ready'] and not result['deployment_changed']
    assert not result['independent_confirmation'] and not result['held_scored']
    assert result['heads'] == 72 and result['primary_optimizer_updates'] == 144000


def test_missing_duplicate_or_undefined_required_view_cannot_be_dropped():
    rows, contexts, cfg, y, env = population()
    with pytest.raises(ValueError):
        api.summarize(rows[:-1], contexts, cfg)
    with pytest.raises(ValueError):
        api.summarize(rows[:-1]+rows[:1], contexts, cfg)
    row = next(r for r in rows if r['mode'] == 'completion_bounds')
    empty = api.base.compact_score(y, env, np.full(3, -1, int))
    row['result']['policies']['physical_neural']['policy'] = empty
    row['result']['recordings']['record']['physical_neural'] = empty
    row['result']['policies']['physical_neural']['known_FDE_gain_floor_percent'] = None
    row['result']['subsets']['hard']['reference_ADE_mass'] = 0.
    row['result']['subsets']['hard']['known_rows'] = 0
    result = api.summarize(rows, contexts, cfg)
    assert not result['fitting_advancement_gate']
    assert result['modes']['completion_bounds']['FDE']['physical_neural']['point'] is None
    assert result['modes']['completion_bounds']['safety']['physical_neural']['undefined_easy_record_views'] == 1
    assert result['modes']['completion_bounds']['subsets']['hard']['macro_status'] == 'not_estimable'


def test_proxy_error_is_explicit_and_inactive_neural_actions_do_not_pollute_physical():
    *_, y, env = population()
    packet = dict(y=y, envelope=env, allowed=np.ones((3, 7), bool), scales=np.ones(3))
    p = np.zeros_like(y)
    result = api.proxy_diagnostics(packet, p, 'physical_only')
    p[:, 6] = 1e10
    assert result == api.proxy_diagnostics(packet, p, 'physical_only')
    assert not result['actual_unknown_errors_observed']
    assert not result['conditional_risk_calibration_certificate']
    assert result['finite_endpoint_proxy_MSE']['unknown']['component_MSE'] == [0., 1., 0., 0., 1.]
    assert np.isnan(packet['y'][-1]).all()
    packet['allowed'][:] = False
    out = api.proxy_diagnostics(packet, p, 'physical_only')
    assert out['finite_endpoint_proxy_MSE']['unknown']['component_MSE'] is None


def test_paired_action_comparison_keeps_unknown_bounds_and_utility_only_matching():
    rollouts = np.zeros((3, 7, 12, 2)); rollouts[:, 0, :, 0] = 1; rollouts[:, 1, :, 0] = 2
    packet = dict(ade=np.tile([4., 3., 2., 1., 4., 3., 2.], (3, 1)),
        floor_ade=np.array([5., 6., np.nan]), floor_rollout=np.zeros((3, 12, 2)), rollouts=rollouts,
        recordings=np.array(['a', 'a', 'b']), frames=np.array([1, 1, 2]), ids=np.arange(3))
    packet['ade'][-1] = np.nan
    out = api.paired(packet, np.zeros(3, int), np.ones(3, int), np.ones(3), np.ones(3))
    assert out['full']['known_delta_mass'] == 2.
    assert out['full']['unknown_disagreement_mass'] == 1.
    assert out['matched_role'] == 'utility_only_not_safety'
    assert out['matched_count'] == 3
