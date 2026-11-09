import numpy as np
import pytest
import torch

from src.world_model import m3w_vehicle_eqmotion as m


def test_context_restores_observed_origins_without_fabricating_neighbors():
    h = np.zeros((3, 8, 2), np.float32)
    h[1, -2, 1] = -1
    origins = np.array([[0., 0.], [3., 2.], [0., 0.]])
    ego, peers, mask = m.inputs(h, origins, np.array([2, 1]))
    np.testing.assert_array_equal(ego.numpy(), h)
    np.testing.assert_array_equal(peers[0, 0, -2].numpy(), [3., 1.])
    assert mask.sum() == 2
    assert not mask[2].any()


def test_bound_caps_peak_from_cv_and_retains_zero_and_small_corrections():
    cv = torch.ones(3, 12, 2)
    proposal = cv.clone()
    proposal[1, :, 0] += .5
    proposal[2, :, 0] += 20
    result = m.bounded(proposal, cv, torch.tensor([1., 1., 2.]))
    torch.testing.assert_close(result[:2], proposal[:2])
    torch.testing.assert_close((result[2]-cv[2]).norm(dim=-1), torch.full((12,), 4.))


def test_objective_ignores_unavailable_target_values_but_charges_disagreement():
    class Fixed:
        def __call__(self, *args):
            return torch.ones(1, 12, 2)
    ego = torch.zeros(1, 8, 2)
    cv = torch.zeros(1, 12, 2)
    y = torch.full_like(cv, float('nan'))
    batch = (ego, ego[:, None], torch.ones(1, 1, dtype=torch.bool), cv, torch.ones(1),
             y, torch.zeros(1, 12, dtype=torch.bool))
    torch.testing.assert_close(m.objective(Fixed(), batch), torch.tensor(2.).sqrt())


def test_real_eqmotion_checkpoint_resume_matches_continuous_fit(monkeypatch, tmp_path):
    from scripts import run_m3w_opendd_vehicle_eqmotion as runner

    torch.set_num_threads(1)
    monkeypatch.setattr(runner.vehicle, 'STEPS', 4)
    monkeypatch.setattr(runner.vehicle, 'SAVE', 2)
    monkeypatch.setattr(runner.vehicle, 'BATCH', 5)
    monkeypatch.setattr(runner, 'SITES', ('rdb1', 'rdb2', 'rdb3'))
    h = np.random.default_rng(7).normal(size=(12, 8, 2))
    x = m.inputs(h, np.zeros((len(h), 2)), np.array([4, 4, 4]))
    cv = torch.zeros(len(h), 12, 2)
    target = torch.arange(len(h), dtype=torch.float32)[:, None, None].expand_as(cv)/8.
    batch = (*x, cv, torch.ones(len(h)), target, torch.ones(len(h), 12, dtype=torch.bool))
    data = {s: {'eq': batch} for s in ('rdb2', 'rdb3')}
    a, b = tmp_path/'continuous', tmp_path/'resumed'
    a.mkdir()
    b.mkdir()
    args = ('synthetic', data, 'rdb1', 17, 'comoving_peers')
    monkeypatch.setattr(runner, 'beat', lambda *a, **kw: None)
    monkeypatch.setattr(runner, 'LOCAL', a)
    runner.fit(*args)
    monkeypatch.setattr(runner, 'LOCAL', b)

    def interrupt(*args, **kw):
        raise InterruptedError('after durable checkpoint')

    monkeypatch.setattr(runner, 'beat', interrupt)
    with pytest.raises(InterruptedError):
        runner.fit(*args)
    name = 'rdb1_17_comoving_peers.pt'
    assert torch.load(b/name, weights_only=True)['step'] == 2
    monkeypatch.setattr(runner, 'beat', lambda *a, **kw: None)
    runner.fit(*args)
    first, resumed = (torch.load(p/name, weights_only=True) for p in (a, b))
    assert first['step'] == resumed['step'] == 4
    assert first['rng'] == resumed['rng']
    for k, v in first['weights'].items():
        torch.testing.assert_close(v, resumed['weights'][k], atol=0, rtol=0)
