import inspect
import numpy as np
import pytest
import torch
from src.world_model import m3w_visual_risk_cost as method


def data():
    rng = np.random.default_rng(17); n = 24
    return dict(x=rng.normal(size=(n, 120)).astype(np.float32),
        moments=np.abs(rng.normal(size=(n, 5))).astype(np.float32),
        weights=np.ones(n), observed=np.arange(n)%4 != 0, envelope=np.ones(n, np.float32)*3,
        context=rng.normal(size=(n, 56)).astype(np.float32), sites=np.repeat(['a', 'b'], n//2), ids=np.arange(n))


def test_image_edges_are_causal_and_agent_fields_are_action_invariant():
    h = np.tile(np.c_[np.arange(8)+88., np.ones(8)*50], (2, 1, 1))
    straight = np.tile(np.c_[np.arange(1, 13), np.zeros(12)], (2, 1, 1))
    a = method.edge_features(h, straight, [100, 100]); b = method.edge_features(h, np.zeros_like(straight), [100, 100])
    np.testing.assert_array_equal(a[:, :6], b[:, :6])
    assert a[0, 6] > b[0, 6] and a[0, 7] < b[0, 7]
    assert list(inspect.signature(method.edge_features).parameters) == ['history', 'rollout', 'native_size']


def test_reference_and_observation_never_see_action_context_even_after_learning():
    d = data(); p = method.prepare(d); torch.manual_seed(17); model = method.Head(method.SETTINGS, p['binding'])
    with torch.no_grad():
        for layer in (model.visual_conditional, model.visual_reference, model.visual_observation):
            torch.nn.init.normal_(layer.weight, std=.1)
        for layer in (model.conditional[-1], model.reference[-1], model.observation[-1]):
            torch.nn.init.normal_(layer.weight, std=.1)
    x = torch.randn(5, 120); c = torch.randn(5, 56); changed_x = x.clone(); changed_c = c.clone()
    changed_x[:, 50:] += 10; changed_c[:, 40:48] += 10; changed_c[:, 54:] -= 10
    a = model(x, torch.ones(5), c); b = model(changed_x, torch.ones(5), changed_c)
    assert torch.equal(a['observation_probability'], b['observation_probability'])
    assert torch.equal(a['conditional'][:, 2:4], b['conditional'][:, 2:4])
    assert not torch.equal(a['conditional'][:, 1], b['conditional'][:, 1])


def test_zero_initialized_context_branches_give_identical_starting_functions():
    d = data(); p = method.prepare(d); torch.manual_seed(17); m = method.Head(method.SETTINGS, p['binding'])
    a = m(p['x'], p['envelope'], p['context']); b = m(p['x'], p['envelope'], torch.zeros_like(p['context']))
    for key in a: assert torch.equal(a[key], b[key])
    assert sum(v.numel() for v in m.parameters()) == 19814


def test_actual_optimizer_resume_matches_direct_and_labels_not_forward_inputs(tmp_path, monkeypatch):
    torch.set_num_threads(2); d = data()
    cfg = dict(method.SETTINGS, steps=4, log_every=1, checkpoint_steps=2, batch_size=8)
    monkeypatch.setattr(method, 'SETTINGS', cfg)
    def run(path, resume=False, stop_at=None):
        return method.fit(d, dict(test='synthetic'), path, arm='motion_geometry', seed=17, resume=resume, stop_at=stop_at)
    a = run(tmp_path/'direct.pt.gz'); run(tmp_path/'resume.pt.gz', stop_at=2)
    b = run(tmp_path/'resume.pt.gz', resume=True); method.exact(a, b)
    assert list(inspect.signature(method.Head.forward).parameters) == ['self', 'x', 'envelope', 'context']
    altered = dict(d, observed=d['observed'].astype(int))
    with pytest.raises(ValueError): method.prepare(altered)
