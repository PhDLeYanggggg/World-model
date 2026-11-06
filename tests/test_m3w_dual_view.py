import numpy as np
import pytest
import torch

from src.world_model.m3w_context_conditioning import INPUT_KEYS, condition_inputs
from src.world_model.m3w_dual_view import DualViewForecaster, aligned_node_features, pack_inputs
from src.world_model.m3w_european_source_forecast import SourceForecaster, baseline_torch
from src.world_model.m3w_native_forecast import pack_geometry
from src.training import m3w_dual_view as training
from tests.test_m3w_visual_forecast import ARCH, geometry
from tests.test_camera_factor_identifiability import request
from tests.test_m3w_camera_frame import flows
from src.world_model.m3w_camera_frame import query_maps


def batch(n=4):
    affine = np.eye(3); affine[:2, 2] = [2., -1.]
    maps = query_maps(flows(affine), 7)['past']
    return pack_inputs(geometry(n), np.zeros((n, 2)), np.broadcast_to(maps, (n, 8, 3, 3)))


def test_frozen_control_initialization_rng_and_actual_core_updates_match():
    torch.manual_seed(17); old = SourceForecaster(ARCH, 1); rng = torch.get_rng_state().clone()
    torch.manual_seed(17); new = DualViewForecaster(ARCH, 1, 'raw_control')
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters()) == 1792
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for m in (old, new)]
    for _ in range(3):
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        a, b = old(pack_geometry(geometry())), new(batch())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward(); b.square().mean().backward()
        grads = [torch.nn.utils.clip_grad_norm_(m.parameters(), 5.) for m in (old, new)]
        assert torch.equal(*grads)
        for opt in optimizers: opt.step()
        for key, value in old.state_dict().items(): assert torch.equal(value, new.state_dict()[key])
    from scripts.run_m3w_moving_source_forecast import exact
    exact(optimizers[0].state_dict()['state'], optimizers[1].state_dict()['state'])
    assert all(p.grad is None for p in new.node_encoder.parameters())


def test_matched_learning_arms_start_at_original_and_retain_native_geometry():
    models = []
    for arm in ('native_node', 'camera_node'):
        torch.manual_seed(17); models.append(DualViewForecaster(ARCH, 1, arm))
    torch.manual_seed(17); original = SourceForecaster(ARCH, 1)
    for model in models:
        torch.testing.assert_close(model(batch()), original(pack_geometry(geometry())), atol=0, rtol=0)
        with torch.no_grad(): model.node_encoder[-1].weight.fill_(.2)
    captured = []
    for model in models:
        hook = model.predictor.predictor.core.register_forward_pre_hook(lambda module, args: captured.append((args[0].clone(), args[1].clone())))
        result = model(batch()); result.square().mean().backward(); hook.remove()
        assert result.shape == (4, 12, 2) and torch.isfinite(result).all()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.node_encoder.parameters())
    torch.testing.assert_close(captured[0][1], captured[1][1], atol=0, rtol=0)
    assert not torch.equal(captured[0][0], captured[1][0])


def test_two_views_preserve_information_erased_by_past_only_model():
    views = []
    for camera in ([0., 0.], [2., -1.]):
        x, _ = request(camera); x.pop('predicted_camera_maps')
        safe = {k: x[k] for k in INPUT_KEYS}; safe['baseline'] = baseline_torch(safe['history'], 1)
        conditioned, scale = condition_inputs(safe)
        torch.manual_seed(17); model = DualViewForecaster(ARCH, 1, 'camera_node')
        context = model.predictor.predictor.prepare_context(conditioned)
        features = aligned_node_features(context, scale, x['observed_camera_maps'])
        views.append((features, scale))
    assert not torch.equal(views[0][0][..., :16], views[1][0][..., :16])
    # The same world velocity is divided by different native-history scales.
    restored = [torch.sinh(features[:, 0, 16:32])*scale[:, None] for features, scale in views]
    torch.testing.assert_close(restored[0], restored[1], atol=1e-6, rtol=1e-6)


def test_future_fields_and_future_camera_restoration_rejected():
    model = DualViewForecaster(ARCH, 1, 'camera_node'); x = batch()
    for name in ('future_endpoint', 'future_mask', 'predicted_camera_maps'):
        with pytest.raises(ValueError): model(dict(x, **{name: torch.zeros(4, 12, 2)}))


class Tiny(torch.nn.Module):
    def __init__(self, fail_at=None, stop=None):
        super().__init__(); self.w = torch.nn.Parameter(torch.randn(12, 2)*.01)
        self.calls = 0; self.fail_at = fail_at; self.stop = stop
    def forward(self, inputs):
        self.calls += 1
        if self.calls == self.fail_at: raise RuntimeError('Synthetic partial-forward error')
        if self.stop is not None and self.calls == 2: self.stop.request()
        return inputs['baseline']+self.w


def problem():
    n = 12; valid = np.ones((n, 12), bool); valid[2] = False
    target = np.ones((n, 12, 2), np.float32); target[2] = np.nan
    data = dict(geometry=geometry(n), target=target, valid=valid, sites=np.repeat(['a', 'b', 'c'], 4),
        origin=np.zeros((n, 2)), camera_query_index=np.zeros(n, np.int32),
        camera_past=np.broadcast_to(np.eye(3), (1, 8, 3, 3)))
    fold = dict(train_ids=np.arange(8), held_ids=np.arange(8, 12), groups=[np.arange(4), np.arange(4, 8)],
        factors=np.r_[np.ones(8), np.zeros(4)])
    settings = dict(steps=4, batch_size=6, learning_rate=.001, weight_decay=.0001,
        minimum_lr_ratio=.01, checkpoint_every=2, heartbeat_every=1, gradient_clip=5.)
    return data, fold, settings


def run_tiny(tmp_path, name, resume=False, fail_at=None, stop=None, stop_at=None):
    data, fold, settings = problem(); torch.manual_seed(17); model = Tiny(fail_at, stop)
    training.fit(model, data, fold, seed=17, settings=settings, identity={'fixed': True}, directory=tmp_path/name,
        resume=resume, stop_at=stop_at, guard=stop.check if stop is not None else lambda: None)
    return torch.load(tmp_path/name/'checkpoint.pt', weights_only=False)


def test_partial_forward_failure_never_overwrites_coherent_sampler_boundary(tmp_path):
    direct = run_tiny(tmp_path, 'direct')
    with pytest.raises(RuntimeError, match='partial-forward'):
        run_tiny(tmp_path, 'failed', fail_at=2)
    saved = torch.load(tmp_path/'failed/checkpoint.pt', weights_only=False)
    assert saved['step'] == 0 and saved['draws'].sum() == 0 and saved['committed_update_boundary']
    from scripts.run_m3w_moving_source_forecast import same_scientific_checkpoint
    same_scientific_checkpoint(direct, run_tiny(tmp_path, 'failed', resume=True))


def test_mid_forward_stop_request_commits_update_then_resumes_exactly(tmp_path):
    direct = run_tiny(tmp_path, 'direct'); stop = training.DeferredStop()
    with pytest.raises(KeyboardInterrupt): run_tiny(tmp_path, 'stopped', stop=stop)
    saved = torch.load(tmp_path/'stopped/checkpoint.pt', weights_only=False)
    assert saved['step'] == 2 and saved['draws'].sum() == 12 and saved['committed_update_boundary']
    from scripts.run_m3w_moving_source_forecast import same_scientific_checkpoint
    same_scientific_checkpoint(direct, run_tiny(tmp_path, 'stopped', resume=True))


def test_input_packing_ignores_targets_and_requires_no_future_operator():
    data, _, _ = problem(); ids = np.arange(4)
    a = training.batch_inputs(data, ids); data['target'][:] = 1e9
    b = training.batch_inputs(data, ids)
    assert set(a) == INPUT_KEYS|{'observed_camera_maps'}
    for key in a: torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
