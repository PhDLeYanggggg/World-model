import pytest
import torch

from src.world_model import m3w_task_risk_objective as api


def inputs():
    p = torch.tensor([[2., .01, 5., 2., .001], [.3, 1., 4., 2., .8]], requires_grad=True)
    y = torch.tensor([[1., 0., 5., 2., 0.], [0., 1., 4., 2., 1.]])
    return p, y, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool), torch.ones(8)


def test_soft_action_is_causal_and_masked():
    p, y, seg, eligible, rms = inputs()
    a = api.soft_action(p, eligible, rms, 1.)
    assert a.shape == (2,) and ((a > 0) & (a < 1)).all()
    assert a[0] > a[1]
    assert api.soft_action(p, torch.tensor([False, True]), rms, 1.)[0] == 0
    changed_labels = y.flip(0)
    assert torch.equal(api.terms(p, y, seg, 2, eligible, rms, 1.)['action'],
                       api.terms(p, changed_labels, seg, 2, eligible, rms, 1.)['action'])


def test_query_weighted_regret_and_risk_match_scalar_calculation():
    p, y, seg, eligible, rms = inputs()
    r = api.terms(p, y, seg, 2, eligible, rms, 1.)
    a = r['action'].detach()
    assert float(r['regret'].detach()) == pytest.approx(float(((1-a)*y[:, 0]+a*y[:, 1]).mean()))
    assert float(r['risk_excess'][0].detach()) == pytest.approx(float((a*(y[:, 1]-.02*y[:, 2])).mean()))
    assert float(r['risk_excess'][1].detach()) == pytest.approx(float((a*(y[:, 4]-.02*y[:, 3])).mean()))
    api.objective(r, torch.tensor([1., 2.])).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_duplicating_agents_does_not_change_query_weight():
    p, y, seg, eligible, rms = inputs()
    a = api.terms(p, y, seg, 2, eligible, rms, 1.)
    ix = torch.tensor([0, 0, 0, 1])
    b = api.terms(p[ix], y[ix], seg[ix], 2, eligible[ix], rms, 1.)
    assert torch.allclose(a['regret'], b['regret'])
    assert torch.allclose(a['risk_excess'], b['risk_excess'])


def test_no_future_label_gradients_or_unknown_imputation():
    p, y, seg, eligible, rms = inputs()
    y.requires_grad_(True)
    with pytest.raises(ValueError): api.terms(p, y, seg, 2, eligible, rms, 1.)
    y = y.detach(); y[1] = float('nan')
    with pytest.raises(ValueError): api.terms(p, y, seg, 2, eligible, rms, 1.)


def test_dual_ascent_sign_projection_and_detachment():
    old = torch.tensor([.1, .1])
    excess = torch.tensor([1., -1.], requires_grad=True)
    value = api.update_dual(old, excess, .5, 2.)
    assert torch.equal(value, torch.tensor([.6, 0.]))
    assert not value.requires_grad
    assert torch.equal(api.update_dual(value, torch.tensor([100., 100.]), 1., 2.), torch.tensor([2., 2.]))


def test_gradient_matches_finite_differences():
    p, y, seg, eligible, rms = inputs()
    p, y, rms = p.double(), y.double(), rms.double()
    def loss(value):
        return api.objective(api.terms(value, y, seg, 2, eligible, rms, .7), torch.tensor([2., 3.]))
    assert torch.autograd.gradcheck(loss, (p,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize('change', ['temperature', 'rms', 'segments', 'eligibility'])
def test_invalid_causal_gate_fails(change):
    p, y, seg, eligible, rms = inputs(); t = 1.
    if change == 'temperature': t = 0.
    elif change == 'rms': rms[0] = 0
    elif change == 'segments': seg = torch.tensor([0, 0])
    else: eligible = eligible.float()
    with pytest.raises(ValueError): api.terms(p, y, seg, 2, eligible, rms, t)
