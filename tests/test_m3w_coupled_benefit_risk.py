import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_coupled_benefit_risk as api
from tests.test_m3w_selected_query_critic import fixture, settings


def data():
    c, y = fixture()
    return api.parent.training_data(c, y)


def test_anchor_exact_and_shared_capacity():
    d = data(); args = api.parent.batch(d['causal'], np.arange(8), np.repeat(np.arange(4), 2), 4)
    models = [api.CoupledCritic(4, 8, 17, a) for a in api.ARMS]
    api.parent.exact(models[0].state_dict(), models[1].state_dict())
    for m in models:
        torch.testing.assert_close(m(*args), args[1], rtol=0, atol=0)


def test_released_benefit_expands_harm_capacity_but_not_reference_budget():
    d = data(); args = api.parent.batch(d['causal'], np.arange(8), np.repeat(np.arange(4), 2), 4)
    m = api.CoupledCritic(4, 8, 17, 'released_benefit')
    with torch.no_grad():
        m.benefit_output.bias.fill_(-1.); m.output.bias[0] = .4
    p = m(*args)
    assert (p[:, :, 0] < args[1][:, :, 0]).all()
    assert (p[:, :, 1] > args[1][:, :, 1]).all()
    torch.testing.assert_close(p[:, :, [2, 3]], args[1][:, :, [2, 3]], rtol=0, atol=0)
    assert (p >= 0).all() and (p[:, :, :2].sum(-1) <= args[2]+1e-12).all()
    assert (p[:, :, 4] <= p[:, :, 1]).all()


def test_old_zero_capacity_can_be_released_without_nan_or_future_inputs():
    d = data(); c = copy.deepcopy(d['causal']); c['prediction'][:, :, :2] = [1., 0.]; c['prediction'][:, :, 4] = 0.
    args = api.parent.batch(c, np.arange(8), np.repeat(np.arange(4), 2), 4)
    m = api.CoupledCritic(4, 8, 17, 'released_benefit')
    with torch.no_grad():
        m.benefit_output.bias.fill_(-.5); m.output.bias.fill_(.3)
    p = m(*args); assert torch.isfinite(p).all() and (p[:, :, 1] > 0).all()
    assert (p[:, :, 4] > 0).all() and (p[:, :, 4] <= p[:, :, 1]).all()
    p[:, :, [0, 1, 4]].sum().backward()
    assert torch.isfinite(m.benefit_output.weight.grad).all()


def test_protected_real_updates_reproduce_old_unit_corrected_model_exactly(tmp_path):
    d = data(); old = api.units.fit(d, normalization='selected_query', arm='selected_set', seed=17,
        settings=settings(), identity={}, path=tmp_path/'old.pt.gz')
    new = api.fit(d, mode='protected_benefit', seed=17, settings=settings(), identity={}, path=tmp_path/'new.pt.gz')
    api.assert_old_control(new, old)
    np.testing.assert_array_equal(api.predict(new, d['causal']), api.parent.predict(old, d['causal']))


@pytest.mark.parametrize('mode', api.ARMS)
def test_real_training_resume_exact_unknowns_unsupervised_and_references_fixed(tmp_path, mode):
    d = data(); kw = dict(mode=mode, seed=17, settings=settings(), identity={})
    direct = api.fit(d, path=tmp_path/'direct.pt.gz', **kw)
    api.fit(d, path=tmp_path/'resume.pt.gz', stop_at=2, **kw)
    recovered = api.fit(d, path=tmp_path/'resume.pt.gz', resume=True, **kw)
    for k in direct:
        if k != 'seconds': api.parent.exact(direct[k], recovered[k])
    p = api.predict(direct, d['causal'])
    np.testing.assert_array_equal(p[:, :, [2, 3]], d['causal']['prediction'][:, :, [2, 3]])
    assert direct['unknown_rows_with_loss'] == 0 and direct['unknown_rows_drawn'] > 0
    if mode == 'released_benefit': assert not np.array_equal(p[:, :, 0], d['causal']['prediction'][:, :, 0])
    with pytest.raises(ValueError): api.predict(direct, dict(d['causal'], future_endpoint=np.zeros((8, 2))))


def test_matched_initialization_draws_and_fitting_units(tmp_path):
    d = data(); states = [api.fit(d, mode=a, seed=17, settings=settings(), identity={},
        path=tmp_path/(a+'.pt.gz')) for a in api.ARMS]
    api.assert_matched(*states)
    assert states[0]['parameters'] == states[1]['parameters']
