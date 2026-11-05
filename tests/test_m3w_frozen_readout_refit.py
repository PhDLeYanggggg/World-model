import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from src.world_model import m3w_frozen_readout_refit as api
from tests.test_m3w_direct_tail_risk import data


def test_weights_equal_repeated_query_objective_and_keep_empty_mass():
    d = data(); c = d['causal']; c['allowed'][1] = False
    weights = api.query_weights(c, np.array([3, 1]))
    np.testing.assert_allclose(weights, [.75, 0, .125, .125])
    prediction = torch.arange(56).reshape(4, 7, 2).float()/50
    ix = np.array([0, 1, 0, 1, 0, 1, 2, 3]); seg = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    # The objective indexes supervision, while receiving the duplicated batch outputs.
    actual = api.parent.objective(prediction[ix], d, ix, seg, 4, 'mean')['total'].item()
    target = api.parent.signed(d['truth_by_arm']['bound_aware'])/torch.from_numpy(api.parent.normalizer(d['preprocess']))
    expected = float(np.sum(weights*((prediction[:, 6]-target[:, 6])**2).mean(1).numpy()))
    assert actual == pytest.approx(expected, rel=1e-6)
    c['allowed'][2:] = False
    assert api.query_weights(c, np.array([3, 1])).sum() == .75
    with pytest.raises(ValueError, match='integer query'):
        api.query_weights(c, np.array([0, 1]))


def test_svd_recovers_weighted_affine_fit_and_rank_deficiency():
    rng = np.random.default_rng(3)
    x = np.column_stack((rng.normal(size=(80, 3)), np.ones(80)))
    truth = rng.normal(size=(4, 2)); y = x@truth; w = np.arange(1, 81)/3240
    beta, audit = api.solve(x, y, w, np.zeros_like(truth))
    np.testing.assert_allclose(beta, truth, atol=1e-12)
    assert audit['solved_float64_MSE'] < 1e-20 and audit['relative_normal_residual'] < 1e-12
    xx = np.column_stack((x, x[:, 0])); old = np.zeros((5, 2))
    beta, rank = api.solve(xx, y, w, old)
    assert rank['retained_rank'] == 4
    np.testing.assert_allclose(xx@beta, y, atol=1e-12)


def test_real_readout_refit_freezes_encoder_and_replays(tmp_path):
    torch.set_num_threads(1)
    d = data(); rng = np.random.default_rng(19)
    d['causal']['x'][..., :380] = rng.normal(size=(4, 7, 380))*.1
    d['z'] = torch.from_numpy(d['causal']['x'].copy())
    d['truth_by_arm']['bound_aware'][2:, :, 1] = .3
    settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=2000,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1000, checkpoint_every=2000)
    warm = api.parent.fit(d, arm='mean', seed=17, settings=settings, identity={}, path=tmp_path/'warm.pt.gz')
    args = dict(identity={'fixed': True})
    first = api.refit(d, warm, **args, path=tmp_path/'first.pt.gz')
    repeat = api.refit(d, warm, **args, path=tmp_path/'repeat.pt.gz')
    api.exact_refit(first, repeat)
    resumed = api.refit(d, warm, **args, path=tmp_path/'first.pt.gz', resume=True)
    api.exact_refit(first, resumed)
    assert first['diagnostics']['refit_actual_float32_MSE'] <= first['diagnostics']['original_actual_float32_MSE']+1e-5
    assert first['new_encoder_updates'] == first['new_adam_updates'] == 0
    assert first['optimizer_state'] is None
    for key in warm['model']:
        if key.startswith('encoder.'):
            assert torch.equal(warm['model'][key], first['model'][key])
    np.testing.assert_array_equal(api.predict(first, d['causal']), api.predict(repeat, d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.extract(warm, dict(d['causal'], future_endpoint=np.zeros((4, 2))))
    before = api.extract(warm, d['causal'])['design']
    d['truth_by_arm']['bound_aware'][:] = 999
    np.testing.assert_array_equal(before, api.extract(warm, d['causal'])['design'])
    with pytest.raises(AssertionError):
        api.refit(d, warm, **args, path=tmp_path/'leaky.pt.gz')


def test_runner_freezes_three_heads_and_decides_before_labels():
    from scripts import run_m3w_moving_readout_refit as run
    config = run.source.read(run.CONFIG)
    assert config['new_readout_fits'] == 3 and config['new_adam_updates'] == 0
    assert config['risk_budget'] == config['easy_net_budget'] == .02
    assert not config['threshold_search'] and not config['identity_certification']
    tree = ast.parse(Path(run.__file__).read_text())
    functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "freeze['complete_readouts'] != 3" in functions['evaluate']
    assert functions['evaluate'].index('parent.api.choose(') < functions['evaluate'].index("data['target_eval']")


def test_action_flow_partitions_proposals_and_does_not_label_unknown():
    from scripts import run_m3w_moving_readout_refit as run
    moments = np.ones((5, 7, 5)); moments[-1] = 1e9
    truth = dict(moments=moments, observed_future=np.array([True, True, True, True, False]))
    result = run.flow(truth, np.array([6, 6, 6, -1, 6]), np.array([6, 6, -1, -1, -1]),
        np.array([6, -1, 6, -1, -1]))
    assert result['retained']['known_rows'] == 1
    assert result['newly_rejected']['known_rows'] == 1
    assert result['newly_admitted']['known_rows'] == 1
    assert result['still_rejected']['unknown_rows'] == 1
    assert result['still_rejected']['known_moments'] == [0.]*5
