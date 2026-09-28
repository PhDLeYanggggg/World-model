import numpy as np
import pytest
import torch
from src.world_model import m3w_easy_hurdle as api
from src.world_model.m3w_easy_gradient_diagnostic import geometry, gradients, fitting_signal


def test_conflict_geometry_and_undefined_zero_gradient():
    v = geometry(dict(marginal=torch.tensor([1., 0.]), occurrence=torch.tensor([-3., 0.]), conditional=torch.tensor([0., 1.])))
    assert v['risk_cosine']['occurrence'] == -1
    assert v['risk_projection']['supervised'] == -2
    assert v['auxiliary_to_risk_norm'] == pytest.approx(10**.5)
    zero = geometry({k: torch.zeros(2) for k in ('marginal', 'occurrence', 'conditional')})
    assert zero['auxiliary_to_risk_norm'] is None
    assert all(c is None for c in zero['risk_cosine'].values())


def test_loss_gradients_leave_model_unchanged():
    model = api.initialize(4, 8, 17)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.arange(16).reshape(4, 4).float()/10
    y = torch.tensor([[1., .3, .01], [0., 1., .2], [1., .5, .1], [0., .9, .3]])
    row = gradients(model, x, torch.ones(4), y, torch.tensor([0, 0, 1, 1]), 2)
    assert set(row['gradients']) == {'all', 'shared', 'output'}
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(p.grad is None for p in model.parameters())


def test_directional_gradient_agrees_with_finite_difference():
    model = api.initialize(4, 8, 17).double()
    x = torch.arange(16).reshape(4, 4).double()/10
    env = torch.ones(4, dtype=torch.float64)
    y = torch.tensor([[1., .3, .01], [0., 1., .2], [1., .5, .1], [0., .9, .3]], dtype=torch.float64)
    segments = torch.tensor([0, 0, 1, 1]); params = list(model.parameters())
    losses = api.losses(model(x, env), y, segments, 2)
    g = torch.autograd.grad(losses['marginal'], params, retain_graph=True)
    aux = torch.autograd.grad(losses['occurrence']+losses['conditional'], params)
    exact = sum(float((a*b).sum()) for a, b in zip(g, aux))
    state = {k: v.clone() for k, v in model.state_dict().items()}; values = []
    for sign in (-1, 1):
        model.load_state_dict(state)
        with torch.no_grad():
            for p, direction in zip(params, aux):
                p.add_(sign*1e-6*direction)
        values.append(float(api.losses(model(x, env), y, segments, 2)['marginal'].detach()))
    assert (values[1]-values[0])/2e-6 == pytest.approx(exact, rel=1e-6, abs=1e-8)


def test_fitting_signal_is_query_balanced_and_excludes_unknown():
    y = np.array([[1., 1., .01], [0., 2., .1], [1., 1., .1], [np.nan]*3])
    out = fitting_signal(y, [np.array([0, 1]), np.array([2])], {'a': [0, 1]})['a']
    assert out['easy_probability'] == .75
    assert out['easy_reference_mass'] == .75
    assert out['easy_harm_mass'] == pytest.approx(.0525)
    with pytest.raises(ValueError):
        fitting_signal(y, [np.array([3])], {'a': [0]})


def test_actual_source_group_layout_is_mapped_to_source_names():
    y = np.array([[1., 1., .01], [0., 2., .1], [1., 1., .1], [np.nan]*3])
    sites = np.array(['a', 'a', 'b', 'b'])
    groups, sources, _ = api.sampling.query_groups(sites, np.repeat('r', 4),
                                                   [1, 1, 1, 2], np.isfinite(y).all(1))
    out = fitting_signal(y, groups, dict(zip(sorted(set(sites)), sources)))
    assert set(out) == {'a', 'b'}
    assert out['a']['easy_probability'] == .5
    assert out['b']['easy_probability'] == 1
