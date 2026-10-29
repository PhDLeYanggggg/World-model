import copy

import pytest
import torch

from src.world_model import m3w_cost_sensitive_policy as api
from tests.test_m3w_direct_policy import fixture, settings


def test_cost_surrogate_recovers_gradient_for_confident_wrong_action():
    z = torch.tensor([30., -30., 30., -30., 0.], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([2., -2., -2., 2., 0.], dtype=torch.float64)
    losses = api.cost_surrogate(z, c, 1.)
    gradient, = torch.autograd.grad(losses.sum(), z)
    assert gradient[0] > 1.99 and gradient[1] < -1.99
    assert abs(gradient[2]) < 1e-10 and abs(gradient[3]) < 1e-10 and gradient[4] == 0
    assert torch.isfinite(losses).all()


def test_cost_surrogate_is_cost_weighted_not_unweighted_hard_label():
    z = torch.zeros(3, requires_grad=True)
    c = torch.tensor([.1, 10., -2.])
    g, = torch.autograd.grad(api.cost_surrogate(z,c,.5).sum(),z)
    torch.testing.assert_close(g, c)
    with pytest.raises(ValueError): api.cost_surrogate(z,c,0.)


def test_chunked_gradient_and_hard_statistics_match_dense_calculation():
    data = api.base.prepare(*fixture()); loc = api.whole.layout(data)
    model = api.base.Policy(data['z'].shape[1],8,17)
    dual = torch.full((len(data['groups']),2),.7)
    initial = copy.deepcopy(model.state_dict())
    stats = api.full_pass(model,data,loc,dual,.5,2,True)
    actual = torch.cat([p.grad.flatten() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    ix,rec = loc['index'],loc['record']
    z = model(data['z'][ix],data['anchor'][ix])
    c = -data['gain'][ix]+(dual[rec]*data['risk'][ix]/data['record_scales'][rec]).sum(1)
    reference = (api.cost_surrogate(z,c,.5)*loc['weight']).sum()
    reference.backward()
    expected = torch.cat([p.grad.flatten() for p in model.parameters()])
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    assert stats['surrogate_objective'] == pytest.approx(float(reference.detach()))
    old = api.whole.full_pass(model,data,loc,dual,.5,2)
    for key in ('gain','risk','action'):
        torch.testing.assert_close(stats[key],old[key])
    api.base.core.exact(initial,model.state_dict())


def test_exact_resume_does_not_change_training_or_inputs(tmp_path):
    d = api.base.prepare(*fixture())
    kw = dict(settings={**settings(),'chunk_size':2},seed=17,identity={'test':True})
    full = api.fit(d,path=tmp_path/'full.pt.gz',**kw)
    api.fit(d,path=tmp_path/'resume.pt.gz',stop_at=2,**kw)
    resumed = api.fit(d,path=tmp_path/'resume.pt.gz',resume=True,**kw)
    for k in full:
        if k != 'seconds': api.base.core.exact(full[k],resumed[k])
    assert any(not torch.equal(v,full['initial_model'][k]) for k,v in full['model'].items())
    assert not full['held_used_for_training'] and not full['new_forecaster']
    assert full['optimizer_steps'] == 6
