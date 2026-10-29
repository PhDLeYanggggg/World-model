import numpy as np
import pytest
import torch
from src.world_model.m3w_european_source_forecast import SourceForecaster, fit_design
from src.world_model.m3w_native_forecast import masked_objective, fit_trial as native_fit, pack_geometry
from src.world_model.m3w_reference_harm_forecast import (
    reference_harm_objective, fit_trial, assert_training_equivalent)


def test_harm_loss_exact_arithmetic_and_detached_references():
    p = torch.tensor([[[3., 0.]], [[1., 0.]]], requires_grad=True)
    y = torch.zeros_like(p)
    valid = torch.ones(2, 1, dtype=torch.bool)
    f = torch.ones(2)
    selected = torch.tensor([[[2., 0.]], [[2., 0.]]], requires_grad=True)
    cv = torch.tensor([[[1., 0.]], [[2., 0.]]], requires_grad=True)
    loss, ade, terms = reference_harm_objective(p, y, valid, f, selected, cv,
                                               selected_weight=2., cv_weight=2.)
    assert loss.item() == 5.
    assert terms['selected_positive_harm'].item() == .5
    assert terms['cv_positive_harm'].item() == 1.
    loss.backward()
    torch.testing.assert_close(p.grad[:, 0, 0], torch.tensor([2.5, .5]))
    assert selected.grad is None and cv.grad is None


def test_masked_unknown_labels_do_not_enter_harm_or_gradient():
    p = torch.ones(2, 12, 2, requires_grad=True)
    y = torch.full_like(p, float('nan'))
    y[0, 0] = 0
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, 0] = True
    loss, ade, _ = reference_harm_objective(p, y, mask, torch.ones(2), p.detach()*0, p.detach()*0,
                                           selected_weight=2., cv_weight=2.)
    assert ade[1].item() == 0 and torch.isfinite(loss)
    loss.backward()
    assert not p.grad[1].any() and not p.grad[0, 1:].any()


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf')])
def test_invalid_harm_weights_rejected(bad):
    x = torch.zeros(1, 12, 2)
    with pytest.raises(ValueError):
        reference_harm_objective(x, x, torch.ones(1, 12, dtype=torch.bool), torch.ones(1), x, x,
                                 selected_weight=bad, cv_weight=2.)


def fixture():
    g = np.zeros((12, 476), np.float32)
    g[:, :16] = np.tile(np.arange(-7, 1).repeat(2), (12, 1))
    g[:, 16:24] = np.arange(-7, 1)/12
    g[:, 332:356] = np.arange(1, 13).repeat(2)
    data = dict(geometry=g, target=g[:, 332:356].reshape(-1, 12, 2).copy()+1,
                valid=np.ones((12, 12), bool), sites=np.repeat(['a', 'b', 'c'], 4))
    design = fit_design(data, ['a', 'b'], np.tile([3., 2., 1., 2., 2., 2.], (12, 1)))
    arch = dict(width=16, heads=2, layers=1, neighbor_policy='complete_aligned_history',
                input_conditioning='observed_joint_max_norm', output_parameterization='motion_bounded')
    settings = dict(steps=6, batch_size=4, learning_rate=.001, minimum_lr_ratio=.01,
                    weight_decay=.0001, gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    return data, design, arch, settings


def test_exact_neutral_and_interrupted_training(tmp_path):
    torch.set_num_threads(4)
    data, design, arch, settings = fixture()
    for mode in ('native', 'neutral', 'harm', 'resumed'):
        torch.manual_seed(17)
        model = SourceForecaster(arch, 2)
        kwargs = dict(seed=17, settings=settings, identity={'fixture':True},
                      directory=tmp_path/mode, heartbeat=lambda **kw:None)
        if mode == 'native':
            native_fit(model, data, design, **kwargs)
        else:
            weights = dict(selected_weight=0., cv_weight=0.) if mode == 'neutral' else {}
            fit_trial(model, data, design, **kwargs, **weights, stop_at=3 if mode == 'resumed' else None)
            if mode == 'resumed':
                fit_trial(SourceForecaster(arch, 2), data, design, **kwargs, resume=True)
    load = lambda name:torch.load(tmp_path/name/'checkpoint.pt', weights_only=False)
    assert assert_training_equivalent(load('native'), load('neutral'))
    assert assert_training_equivalent(load('harm'), load('resumed'))
    assert load('harm')['draws'][design['held_ids']].sum() == 0


def test_inputs_independent_of_supervision_and_future():
    data, _, arch, _ = fixture()
    torch.manual_seed(17)
    model = SourceForecaster(arch, 2).eval()
    with torch.no_grad():
        before = model(pack_geometry(data['geometry']))
        data['target'][:] = 1e20
        data['valid'][:] = False
        after = model(pack_geometry(data['geometry']))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
