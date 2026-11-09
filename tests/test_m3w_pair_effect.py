import numpy as np
import pytest
import torch

from src.world_model.m3w_pair_effect import PairEffect, observed_effect
from src.world_model.m3w_slate_imitation import Slate


def test_identical_two_edits_can_cancel_a_relative_motion_harm():
    zeros = np.zeros((1, 2, 12, 2))
    offset = np.ones_like(zeros)
    value, common = observed_effect(zeros, offset, zeros, np.ones((1, 2, 12), bool))
    np.testing.assert_allclose(value, [2*np.sqrt(2.)])
    np.testing.assert_array_equal(common, [12])


def test_counterfactual_uses_only_shared_label_steps_and_is_swap_invariant():
    rng = np.random.default_rng(123)
    cv, delta, target = (rng.normal(size=(4, 2, 12, 2)) for _ in range(3))
    valid = rng.uniform(size=(4, 2, 12)) > .4
    a, n = observed_effect(cv, delta, target, valid)
    np.testing.assert_array_equal(n, valid.all(1).sum(1))
    target[~valid] = np.nan
    b, _ = observed_effect(cv[:, ::-1], delta[:, ::-1], target[:, ::-1], valid[:, ::-1])
    np.testing.assert_allclose(a, b)
    delta[:, 1] = 0
    c, _ = observed_effect(cv, delta, target, valid)
    np.testing.assert_allclose(c, 0., atol=1e-15)


def test_no_observations_is_not_a_real_zero_target():
    zeros = np.zeros((1, 2, 12, 2))
    y, common = observed_effect(zeros, zeros, zeros, np.zeros((1, 2, 12), bool))
    assert (y[0], common[0]) == (0., 0)


def test_inference_keeps_additive_predictions_and_only_enables_pairs():
    torch.manual_seed(51)
    slate = Slate()
    net = PairEffect(slate, weight=.3)
    x, geom = torch.randn(3, 2, 45), torch.randn(3, 4)
    mask = torch.tensor([[False, False], [True, False], [True, True]])
    additive, joint = net(x, geom, mask)
    torch.testing.assert_close(additive, slate(x, geom, mask)[0], rtol=0., atol=0.)
    torch.testing.assert_close(additive[:2], joint[:2], rtol=0., atol=0.)
    torch.testing.assert_close(joint, net(x.flip(1), geom, mask.flip(1))[1], rtol=1e-6, atol=1e-7)
    assert not any(p.requires_grad for p in slate.parameters())


def test_donors_omit_held_and_unsupported_sites():
    from scripts.run_m3w_pair_effect import supported_donors

    assert supported_donors({'rdb1': object(), 'rdb2': None, 'rdb3': object()}, 'rdb1') == ('rdb3',)


def test_training_resumes_exactly_from_interrupted_checkpoint(monkeypatch, tmp_path):
    from scripts import run_m3w_pair_effect as run

    def slate(*_):
        return Slate().eval()

    monkeypatch.setattr(run, 'slate_for', slate)
    monkeypatch.setattr(run.study, 'artifact', lambda p: {'path': str(p)})
    monkeypatch.setattr(run, 'STEPS', 100)
    monkeypatch.setattr(run, 'BATCH', 16)
    gen = torch.Generator().manual_seed(418)
    data = {'rdb2': (torch.randn(32, 2, 45, generator=gen), torch.randn(32, 4, generator=gen),
                     torch.randn(32, generator=gen))}
    parent = {'fits': {'rdb1_17': {'checkpoint': {'sha256': 'synthetic'}}}}

    uninterrupted = tmp_path / 'reference'
    uninterrupted.mkdir()
    monkeypatch.setattr(run, 'LOCAL', uninterrupted)
    complete, info = run.fit('test', 'rdb1', 17, parent, data)

    resumed = tmp_path / 'interrupted'
    resumed.mkdir()
    monkeypatch.setattr(run, 'LOCAL', resumed)
    original_heartbeat = run.heartbeat

    def interrupt(**values):
        assert values['step'] == 50
        raise InterruptedError

    monkeypatch.setattr(run, 'heartbeat', interrupt)
    with pytest.raises(InterruptedError):
        run.fit('test', 'rdb1', 17, parent, data)
    monkeypatch.setattr(run, 'heartbeat', original_heartbeat)
    restored, restored_info = run.fit('test', 'rdb1', 17, parent, data)
    torch.testing.assert_close(complete.state_dict(), restored.state_dict(), rtol=0., atol=0.)
    assert info['loss'] == restored_info['loss']
