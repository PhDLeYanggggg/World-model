import ast
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.world_model import m3w_nonlinear_risk_readout as api
from tests.test_m3w_direct_tail_risk import data


def toy_matrix():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(300, 2))
    z = np.column_stack((x, x*x, np.sin(x)))
    y = np.column_stack((x[:, 0]**2+x[:, 1], x[:, 0]*x[:, 1]))
    w = rng.uniform(.5, 1.5, len(x)); w *= .8/w.sum()
    m = dict(score_only=x, latent_score=z, truth=y, offset=np.zeros_like(y),
        residual=y.copy(), weights=w, rows=np.arange(len(x)), known=np.ones(len(x), bool))
    m['hashes'] = {k: api.linear.digest(v) for k, v in m.items()}
    m['stats'] = dict(rows=len(x), empirical_weight_mass=.8)
    return m


def toy_frozen():
    return dict(seed=17, source_sites=['toy_source'], fit_ids=np.arange(300))


def test_actual_boosting_resumes_with_exact_trees_and_outputs(tmp_path):
    m = toy_matrix(); settings = dict(rounds=16, checkpoint_every=4, learning_rate=.1,
        min_samples_leaf=8, max_leaf_nodes=7)
    args = dict(arm='latent_score', settings=settings, identity={'same': True})
    a = api.fit(m, toy_frozen(), **args, path=tmp_path/'a.joblib')
    api.fit(m, toy_frozen(), **args, path=tmp_path/'b.joblib', stop_at=8)
    b = api.fit(m, toy_frozen(), **args, path=tmp_path/'b.joblib', resume=True)
    api.exact_fit(a, b)
    np.testing.assert_array_equal(api.correction(a['models'], m['latent_score']), api.correction(b['models'], m['latent_score']))
    assert a['trace'][-1]['fitting_MSE'] < a['trace'][0]['before_fitting_MSE']
    assert all(model.n_iter_ == 16 and not model.do_early_stopping_ for model in a['models'])
    assert a['raw_weight_mass'] == pytest.approx(.8)
    with pytest.raises(ValueError, match='Explicit'):
        api.fit(m, toy_frozen(), **args, path=tmp_path/'b.joblib')
    altered = copy.deepcopy(m); altered['residual'][0] += 1
    with pytest.raises(AssertionError):
        api.fit(altered, toy_frozen(), **args, path=tmp_path/'b.joblib', resume=True)


def test_feature_ablation_removes_only_latent_columns():
    h = dict(prediction=np.arange(10).reshape(5, 2),
        design=np.column_stack((np.arange(20).reshape(5, 4), np.ones(5))))
    a, b = [api.features(h, arm) for arm in api.ARMS]
    np.testing.assert_array_equal(a, b[:, -2:])
    np.testing.assert_array_equal(b[:, :4], h['design'][:, :4])
    assert a.shape == (5, 2) and b.shape == (5, 6)
    with pytest.raises(ValueError):
        api.features(h, 'future_target')


def test_real_causal_matrix_preserves_exact_targets_and_excludes_future(tmp_path):
    torch.set_num_threads(1); d = data()
    rng = np.random.default_rng(3)
    d['causal']['x'][..., :380] = rng.normal(size=(4, 7, 380))*.1
    d['z'] = torch.from_numpy(d['causal']['x'].copy())
    settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=2000,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1000, checkpoint_every=2000)
    warm = api.linear.parent.fit(d, arm='mean', seed=17, settings=settings, identity={}, path=tmp_path/'warm.pt.gz')
    frozen = api.linear.refit(d, warm, identity={}, path=tmp_path/'frozen.pt.gz')
    m = api.matrix(d, warm, frozen)
    assert m['stats']['score_dimensions'] == 2 and m['stats']['latent_score_dimensions'] == 6
    assert m['stats']['unknown_bound_rows'] == int((~d['observed_future']).sum())
    settings2 = dict(rounds=8, checkpoint_every=4, learning_rate=.05, max_leaf_nodes=3, min_samples_leaf=2)
    state = api.fit(m, frozen, arm='latent_score', settings=settings2, identity={}, path=tmp_path/'head.joblib')
    proposal = np.full(4, 6, int)
    before = api.predict(state, d['causal'], proposal)
    d['truth_by_arm']['bound_aware'][:] = 999
    np.testing.assert_array_equal(before, api.predict(state, d['causal'], proposal))
    with pytest.raises(ValueError, match='future labels'):
        api.predict(state, dict(d['causal'], future_endpoint=np.zeros((4, 2))), proposal)
    with pytest.raises(AssertionError):
        api.matrix(d, warm, frozen)
    original = api.linear.predict(frozen, d['causal'])
    np.testing.assert_array_equal(original, api.predict(state, d['causal'], np.full(4, -1, int)))


def test_runner_freezes_all_arms_and_reads_targets_after_actions():
    from scripts import run_m3w_moving_nonlinear_readout as run
    cfg = run.source.read(run.CONFIG)
    assert cfg['primary_fits'] == 6 and cfg['primary_scalar_boosting_rounds'] == 1536
    assert cfg['risk_budget'] == .02 and not cfg['threshold_search']
    tree = ast.parse(Path(run.__file__).read_text())
    fns = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "freeze['complete_fits'] != 6" in fns['evaluate']
    assert fns['evaluate'].index("chosen[arm + '_veto']") < fns['evaluate'].index("data['target_eval']")
    assert "part_id in (0, 2)" in fns['evaluate']
    assert not cfg['new_encoder_training'] and not cfg['new_H_readout']


def test_score_only_control_can_capture_nonlinearity_without_new_latent(tmp_path):
    m = toy_matrix()
    state = api.fit(m, toy_frozen(), arm='score_only',
        settings=dict(rounds=16, checkpoint_every=4, learning_rate=.1, max_leaf_nodes=7, min_samples_leaf=8),
        identity={}, path=tmp_path/'head.joblib')
    assert state['trace'][-1]['fitting_loss_reduction_percent'] > 25
    assert state['new_encoder_updates'] == state['new_adam_updates'] == 0
