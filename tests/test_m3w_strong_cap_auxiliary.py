import numpy as np
import pytest
import torch
from src.world_model import m3w_strong_cap_auxiliary as m
from tests.test_m3w_membership_cost import fixture


def data():
    x, y, easy, sites, env, pr, cfg = fixture()
    event = np.where(np.isfinite(y).all(1) & (env > 0), y[:, 3] > .02, np.nan)
    return x, y, easy, event, sites, env, pr, cfg


def test_original_control_reconstruction_exact(tmp_path):
    torch.set_num_threads(4)
    x, y, e, t, s, d, pr, cfg = data()
    kw = dict(seed=17, settings=cfg, identity={}, heartbeat=lambda **_: None)
    a, _ = m.fit(x, y, e, t, s, 'outside', d, pr, arm='cost_only', directory=tmp_path/'a', **kw)
    b, _ = m.original.fit(x, y, e, s, d, pr, arm='cost_only', directory=tmp_path/'b', **kw)
    for key in a.state_dict(): assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    np.testing.assert_array_equal(m.predict(a, x, d, pr)[0], m.predict(b, x, d, pr)[0])


@pytest.mark.parametrize('arm', m.ARMS)
def test_resume_and_unknown_rows(tmp_path, arm):
    x, y, e, t, s, d, pr, cfg = data()
    kw = dict(arm=arm, seed=29, settings=cfg, identity={'v': 1}, heartbeat=lambda **_: None)
    a, f = m.fit(x, y, e, t, s, 'outside', d, pr, directory=tmp_path/'a', **kw)
    m.fit(x, y, e, t, s, 'outside', d, pr, directory=tmp_path/'b', stop_at=4, **kw)
    b, _ = m.fit(x, y, e, t, s, 'outside', d, pr, directory=tmp_path/'b', resume=True, **kw)
    assert f['unknown_rows_sampled'] == 0
    for key in a.state_dict(): assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    with pytest.raises(ValueError):
        m.fit(x, y, e, t, s, 'outside', d, pr, directory=tmp_path/'b', resume=True, **dict(kw, seed=43))


def test_auxiliary_mask_does_not_drop_cost_rows():
    costs = torch.tensor([[1., .2, .4, .1], [2., 0., 1., 0.]], requires_grad=True)
    logits = torch.zeros(2, requires_grad=True)
    event = torch.tensor([1., float('nan')])
    loss, _ = m.objective(costs, logits, torch.zeros_like(costs), event, torch.ones(4), 'cap_aux')
    loss.backward()
    assert costs.grad[1, 0] != 0 and logits.grad[1] == 0 and logits.grad[0] != 0
    empty, _ = m.objective(costs, logits, costs.detach(), event*float('nan'), torch.ones(4), 'cap_aux')
    assert torch.isfinite(empty)


def test_held_locality_and_illegal_event_rejected():
    x, y, e, t, s, d, pr, _ = data()
    with pytest.raises(ValueError): m.validate(x, y, e, t, s, s[0], d, pr)
    wrong = t.copy(); wrong[np.flatnonzero(np.isfinite(t))[0]] = .2
    with pytest.raises(ValueError): m.validate(x, y, e, wrong, s, 'outside', d, pr)


def test_matched_arms_keep_draws_and_initialization(tmp_path):
    x, y, e, t, s, d, pr, cfg = data(); states = []
    for arm in m.ARMS:
        m.fit(x, y, e, t, s, 'outside', d, pr, arm=arm, seed=17, settings=cfg,
              identity={}, directory=tmp_path/arm, heartbeat=lambda **_: None)
        states.append(m.restore(tmp_path/arm)[1])
    for state in states[1:]:
        for key in states[0]['initial_model']:
            assert torch.equal(states[0]['initial_model'][key], state['initial_model'][key])
        for key in ('fixed_ids', 'draws', 'loss_scales'):
            np.testing.assert_array_equal(states[0][key], state[key])
        assert torch.equal(states[0]['sampler_rng'], state['sampler_rng'])
