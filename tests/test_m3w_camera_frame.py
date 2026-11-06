import numpy as np
import pytest
import torch

from src.world_model.m3w_camera_frame import query_maps, relative_maps, camera_cv, CameraFrameForecaster, pack_inputs
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry, fit_trial
from src.training.m3w_camera_frame import fit
from tests.test_m3w_visual_forecast import ARCH, geometry


def flows(matrix, frame=7):
    return {f: dict(supported=True, native_affine=np.asarray(matrix).tolist()) for f in range(frame-6, frame+1)}


def test_composition_direction_and_rotating_stationary_point_prediction():
    angle = .06
    a = np.array([[np.cos(angle), -np.sin(angle), 2.], [np.sin(angle), np.cos(angle), -1.], [0., 0., 1.]])
    operators = query_maps(flows(a), 7)
    point = np.array([50., 30., 1.])
    history = np.array([(np.linalg.matrix_power(a, t)@point)[:2] for t in range(-7, 1)])
    aligned = np.einsum('tij,tj->ti', operators['past'][:, :2, :2], history)+operators['past'][:, :2, 2]
    np.testing.assert_allclose(aligned, np.tile(point[:2], (8, 1)), atol=1e-12)
    result = camera_cv(history[None], operators['past'][None], operators['prediction'][None])[0]
    expected = np.array([(np.linalg.matrix_power(a, t)@point)[:2] for t in range(1, 13)])
    np.testing.assert_allclose(result, expected, atol=1e-11)


def test_missing_measurement_preserves_all_rows_with_explicit_identity_imputation():
    records = flows(np.eye(3)); records[3]['supported'] = False
    result = query_maps(records, 7)
    assert not result['active'] and result['supported_pairs'] == 6
    np.testing.assert_array_equal(result['past'], np.broadcast_to(np.eye(3), (8, 3, 3)))
    np.testing.assert_array_equal(result['prediction'], np.broadcast_to(np.eye(3), (12, 3, 3)))
    with pytest.raises(ValueError): query_maps(dict(records, **{'8': records[7]}), 7)


def test_local_relative_operator_matches_native_affine_application():
    matrix = np.array([[1.01, -.02, 2.], [.02, 1.01, 1.], [0., 0., 1.]])
    maps = np.broadcast_to(matrix, (2, 8, 3, 3))
    origins = np.array([[1200., 800.], [70., 85.]])
    local = relative_maps(maps, origins)
    point = np.array([3., -2.])
    expected = (origins+point)@matrix[:2, :2].T+matrix[:2, 2]-origins
    result = np.einsum('nij,j->ni', local[:, 0, :, :2], point)+local[:, 0, :, 2]
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_actual_author_core_identity_camera_function_rng_and_updates_match():
    torch.manual_seed(17); old = SourceForecaster(ARCH, 1); rng = torch.get_rng_state().clone()
    torch.manual_seed(17); new = CameraFrameForecaster(ARCH, 1, 'identity')
    assert torch.equal(rng, torch.get_rng_state())
    g = geometry(); past = np.broadcast_to(np.eye(3), (4, 8, 3, 3)); future = np.broadcast_to(np.eye(3), (4, 12, 3, 3))
    inputs = pack_inputs(g, np.zeros((4, 2)), past, future, 'identity')
    oa, ob = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for m in (old, new)]
    old.train(); new.train()
    for _ in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        x, y = old(pack_geometry(g)), new(inputs)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        x.square().mean().backward(); y.square().mean().backward()
        ga = torch.nn.utils.clip_grad_norm_(old.parameters(), 5.)
        gb = torch.nn.utils.clip_grad_norm_(new.parameters(), 5.)
        assert torch.equal(ga, gb)
        oa.step(); ob.step()
        for name, value in old.state_dict().items(): assert torch.equal(value, new.state_dict()[name])
    from scripts.run_m3w_moving_source_forecast import exact
    exact(oa.state_dict(), ob.state_dict())
    with pytest.raises(ValueError): new(dict(inputs, future_endpoint=torch.zeros(4, 2)))


def test_camera_frame_is_differentiable_and_preserves_parameter_count():
    g = geometry(); a = np.eye(3); a[:2, 2] = [2., -1.]
    operators = query_maps(flows(a), 7)
    past = np.broadcast_to(operators['past'], (4, 8, 3, 3))
    future = np.broadcast_to(operators['prediction'], (4, 12, 3, 3))
    torch.manual_seed(17); model = CameraFrameForecaster(ARCH, 1, 'camera_frame')
    torch.manual_seed(17); original = SourceForecaster(ARCH, 1)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in original.parameters())
    result = model(pack_inputs(g, np.zeros((4, 2)), past, future, 'camera_frame'))
    result.square().mean().backward()
    assert result.shape == (4, 12, 2) and torch.isfinite(result).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.randn(12, 2)*.01)
    def forward(self, inputs): return inputs['baseline']+self.w


def test_training_engine_identity_matches_old_and_interrupted_resume(tmp_path):
    n = 12; g = geometry(n); target = np.ones((n, 12, 2), np.float32)
    valid = np.ones((n, 12), bool); valid[2] = False; target[2] = np.nan
    data = dict(geometry=g, target=target, valid=valid, sites=np.repeat(['a', 'b', 'c'], 4),
        origin=np.zeros((n, 2)), camera_query_index=np.zeros(n, np.int32),
        camera_past=np.broadcast_to(np.eye(3), (1, 8, 3, 3)), camera_prediction=np.broadcast_to(np.eye(3), (1, 12, 3, 3)))
    fold = dict(train_ids=np.arange(8), held_ids=np.arange(8, 12), groups=[np.arange(4), np.arange(4, 8)],
        factors=np.r_[np.ones(8), np.zeros(4)])
    settings = dict(steps=4, batch_size=6, learning_rate=.001, weight_decay=.0001,
        minimum_lr_ratio=.01, checkpoint_every=2, heartbeat_every=1, gradient_clip=5.)
    def run(name, old=False, resume=False, stop_at=None, guard=lambda: None):
        torch.manual_seed(17); model = Tiny()
        options = dict(seed=17, settings=settings, identity={'fixed': True}, directory=tmp_path/name,
            resume=resume, stop_at=stop_at, heartbeat=lambda **_: None)
        if old: fit_trial(model, data, fold, **options)
        else: fit(model, data, fold, arm='identity', guard=guard, **options)
        return torch.load(tmp_path/name/'checkpoint.pt', weights_only=False)
    legacy, direct = run('legacy', old=True), run('direct')
    from scripts.run_m3w_moving_source_forecast import same_scientific_checkpoint
    same_scientific_checkpoint(legacy, direct)
    run('resumed', stop_at=2)
    same_scientific_checkpoint(direct, run('resumed', resume=True))
    calls = [0]
    def interrupt_after_three_updates():
        calls[0] += 1
        if calls[0] == 4:
            raise RuntimeError('Synthetic interruption')
    with pytest.raises(RuntimeError, match='Synthetic interruption'):
        run('interrupted', guard=interrupt_after_three_updates)
    saved = torch.load(tmp_path/'interrupted/checkpoint.pt', weights_only=False)
    assert saved['step'] == 3
    same_scientific_checkpoint(direct, run('interrupted', resume=True))


def test_operator_array_recovery_does_not_overwrite_completed_cache(tmp_path):
    from scripts.run_m3w_moving_camera_frame import atomic_array
    path = tmp_path/'operators.npy'; x = np.broadcast_to(np.eye(3), (8, 3, 3))
    atomic_array(path, x); stamp = path.stat().st_mtime_ns
    atomic_array(path, x.copy()); assert path.stat().st_mtime_ns == stamp
    with pytest.raises(AssertionError): atomic_array(path, x+1)


def test_completed_fit_is_checked_and_reused_without_rewriting(tmp_path, monkeypatch):
    from scripts import run_m3w_moving_camera_frame as runner
    checkpoint = tmp_path/'checkpoint.pt'; path = tmp_path/'fit.json'
    identity = dict(camera_arm='camera_frame', fixed='example')
    torch.save(dict(identity=identity, step=4000), checkpoint)
    path.write_text(runner.source.json.dumps(dict(identity=identity,
        checkpoint={'path': str(checkpoint)}, result=dict(step=4000, complete=True))))
    monkeypatch.setattr(runner, 'checked', lambda _: checkpoint)
    stamp = path.stat().st_mtime_ns
    assert runner.complete_receipt(path, identity)['result']['complete']
    assert path.stat().st_mtime_ns == stamp
    with pytest.raises(ValueError): runner.complete_receipt(path, dict(camera_arm='changed'))
