import numpy as np
import pytest
import torch

from src.world_model.m3w_native_forecast import (
    pack_geometry, fold_design, masked_objective, fit_trial, predict,
)
from src.world_model.m3w_supervised_intervention import build_forecaster


def population(n=12):
    g = np.zeros((n, 476), np.float32)
    g[:, :16] = np.tile(np.column_stack((np.arange(-7, 1)/12, np.zeros(8))).ravel(), (n, 1))
    g[:, 16:24] = np.arange(-7, 1)/12
    g[:, 332:356] = np.tile(np.column_stack((np.arange(1, 13)/12, np.zeros(12))).ravel(), (n, 1))
    y = g[:, 332:356].reshape(n, 12, 2).copy()*1.1
    valid = np.ones((n, 12), bool); valid[1] = False
    y[~valid] = np.nan
    return dict(geometry=g, target=y, valid=valid, scale=np.arange(1, n+1, dtype=float),
                sites=np.repeat(['a', 'b', 'c'], n//3))


def test_prediction_payload_excludes_target_support_and_ids():
    data = population()
    first = pack_geometry(data['geometry'])
    data['target'][:] = 1e20; data['valid'][:] = False
    second = pack_geometry(data['geometry'])
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    assert first['request_mask'].all()
    assert not {'target', 'valid', 'scene', 'agent', 'scale'} & first.keys()


def test_future_neighbor_and_ego_time_rejected():
    g = population()['geometry']
    for column in [16, 166]:
        altered = g.copy(); altered[0, column] = 1; altered[0, 230] = 1
        with pytest.raises(ValueError):
            pack_geometry(altered)


@pytest.mark.parametrize('objective', ['native_coordinate', 'past_normalized'])
def test_held_labels_cannot_change_loss_preprocessing(objective):
    data = population()
    a = fold_design(data, 'c', objective)
    data['target'][8:] *= 10000; data['scale'][8:] *= 999
    b = fold_design(data, 'c', objective)
    for key in ['train_ids', 'factors', 'hard_cut']:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize('objective', ['native_coordinate', 'past_normalized'])
def test_importance_correction_matches_exact_supported_scene_loss(objective):
    data = population(); fold = fold_design(data, 'c', objective)
    ids = fold['train_ids']; mask = torch.tensor(data['valid'][ids])
    pred = torch.tensor(data['geometry'][ids, 332:356].reshape(-1, 12, 2))*.8
    target = torch.tensor(data['target'][ids]); factor = torch.tensor(fold['factors'][ids])
    loss, per = masked_objective(pred, target, mask, factor)
    expected = []
    for site in ['a', 'b']:
        use = data['sites'][ids] == site
        error = np.linalg.norm(pred.numpy()[use]-target.numpy()[use], axis=-1).mean(1)
        cv = np.linalg.norm(data['geometry'][ids[use], 332:356].reshape(-1, 12, 2)-target.numpy()[use], axis=-1).mean(1)
        scale = data['scale'][ids[use]] if objective == 'native_coordinate' else np.ones(use.sum())
        expected.append(np.nanmean(error*scale)/np.nanmean(cv*scale))
    assert float(loss) == pytest.approx(np.mean(expected), rel=2e-6)
    assert per[1] == 0


def test_unknown_targets_cannot_poison_gradients():
    p = torch.ones(2, 12, 2, requires_grad=True)
    y = torch.zeros_like(p); valid = torch.ones(2, 12, dtype=torch.bool)
    valid[0] = False; y[0] = float('nan')
    loss, _ = masked_objective(p, y, valid, torch.ones(2))
    loss.backward()
    assert torch.isfinite(p.grad).all() and not p.grad[0].any()


def test_resume_matches_uninterrupted_real_training(tmp_path):
    torch.set_num_threads(2)
    data = population(); fold = fold_design(data, 'c', 'native_coordinate')
    settings = dict(steps=4, batch_size=4, learning_rate=.001, minimum_lr_ratio=.01,
        weight_decay=.0001, gradient_clip=5, checkpoint_every=2, heartbeat_every=1)
    architecture = dict(width=8, heads=2, layers=1, neighbor_policy='complete_aligned_history',
                        input_conditioning='observed_joint_max_norm', output_parameterization='motion_bounded')
    identity = dict(test='temporary_synthetic_training')
    def run(path, resume=False, stop_at=None):
        torch.manual_seed(17); model = build_forecaster(architecture)
        report = fit_trial(model, data, fold, seed=17, settings=settings, identity=identity,
                           directory=path, resume=resume, stop_at=stop_at, heartbeat=lambda **k: None)
        return model, report
    full, _ = run(tmp_path/'full')
    run(tmp_path/'resume', stop_at=2)
    resumed, result = run(tmp_path/'resume', resume=True)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    assert result['step'] == 4 and result['new_updates'] == 2
    assert np.isfinite(predict(resumed, data, fold['held_ids'])).all()
    altered = dict(identity, test='changed')
    with pytest.raises(ValueError):
        fit_trial(resumed, data, fold, seed=17, settings=settings, identity=altered,
                  directory=tmp_path/'resume', resume=True, heartbeat=lambda **k: None)
