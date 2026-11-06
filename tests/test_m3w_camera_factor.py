import numpy as np
import pytest
import torch

from src.world_model.m3w_camera_factor import CameraFactorForecaster, operator_inputs
from src.world_model.m3w_camera_frame import CameraFrameForecaster, pack_inputs, query_maps
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry
from src.training.m3w_camera_frame import fit
from tests.test_m3w_visual_forecast import ARCH, geometry


def inputs(n=4):
    affine = np.eye(3); affine[:2, 2] = [2., -1.]
    maps = query_maps({f: dict(supported=True, native_affine=affine.tolist()) for f in range(1, 8)}, 7)
    return pack_inputs(geometry(n), np.zeros((n, 2)),
        np.broadcast_to(maps['past'], (n, 8, 3, 3)),
        np.broadcast_to(maps['prediction'], (n, 12, 3, 3)), 'camera_frame')


@pytest.mark.parametrize('arm,past,future', [('identity', False, False),
    ('past_only', True, False), ('future_only', False, True), ('both', True, True)])
def test_exact_factor_switches_never_modify_input_or_add_features(arm, past, future):
    x = inputs(); original = {k: v.clone() for k, v in x.items()}
    selected = operator_inputs(x, arm)
    assert set(selected) == set(x)
    for key, enabled in (('observed_camera_maps', past), ('predicted_camera_maps', future)):
        expected = x[key] if enabled else torch.eye(3)[:2].expand_as(x[key])
        torch.testing.assert_close(selected[key], expected, atol=0, rtol=0)
    for key in x: torch.testing.assert_close(x[key], original[key], atol=0, rtol=0)
    with pytest.raises(ValueError): operator_inputs(dict(x, future_endpoint=torch.zeros(4, 2)), arm)
    with pytest.raises(ValueError): operator_inputs(dict(x, future_mask=torch.ones(4, 12)), arm)


@pytest.mark.parametrize('arm', ['identity', 'both'])
def test_actual_core_extreme_arms_match_old_function_rng_and_updates(arm):
    torch.manual_seed(17)
    old = SourceForecaster(ARCH, 1) if arm == 'identity' else CameraFrameForecaster(ARCH, 1, 'camera_frame')
    rng = torch.get_rng_state().clone()
    torch.manual_seed(17); new = CameraFactorForecaster(ARCH, 1, arm)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    a, b = [torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001) for model in (old, new)]
    x = inputs()
    for _ in range(3):
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
        old_input = pack_geometry(geometry()) if arm == 'identity' else x
        y, z = old(old_input), new(x)
        torch.testing.assert_close(y, z, rtol=0, atol=0)
        y.square().mean().backward(); z.square().mean().backward()
        ga, gb = [torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) for model in (old, new)]
        assert torch.equal(ga, gb)
        a.step(); b.step()
        for key, value in old.state_dict().items(): assert torch.equal(value, new.state_dict()[key])
    from scripts.run_m3w_moving_source_forecast import exact
    exact(a.state_dict(), b.state_dict())


def test_integrated_factor_geometry_distinguishes_double_camera_extrapolation(monkeypatch):
    def cv(self, batch):
        xy = batch['history'][..., :2]
        return xy[:, -1, None]+torch.arange(1, 13)[None, :, None]*(xy[:, -1]-xy[:, -2])[:, None]
    monkeypatch.setattr(SourceForecaster, 'forward', cv)
    x = inputs(1); x['history'][0, :, :2] = torch.arange(-7, 1)[:, None]*torch.tensor([2., -1.])
    native = torch.arange(1, 13)[None, :, None]*torch.tensor([2., -1.])
    for arm, expected in (('identity', native), ('past_only', torch.zeros_like(native)),
            ('future_only', 2*native), ('both', native)):
        actual = CameraFactorForecaster(ARCH, 1, arm)(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('arm', ['past_only', 'future_only'])
def test_actual_new_core_recovery_and_held_source_exclusion(tmp_path, arm):
    n = 12; x = inputs(n); g = geometry(n)
    raw_past = np.broadcast_to(np.eye(3), (1, 8, 3, 3)).copy()
    raw_future = np.broadcast_to(np.eye(3), (1, 12, 3, 3)).copy()
    raw_past[:, :, :2] = x['observed_camera_maps'][0].numpy()
    raw_future[:, :, :2] = x['predicted_camera_maps'][0].numpy()
    target = np.ones((n, 12, 2), np.float32); valid = np.ones((n, 12), bool)
    valid[2] = False; target[2] = np.nan
    data = dict(geometry=g, target=target, valid=valid, sites=np.repeat(['a', 'b', 'c'], 4),
        origin=np.zeros((n, 2)), camera_query_index=np.zeros(n, np.int32),
        camera_past=raw_past, camera_prediction=raw_future)
    fold = dict(train_ids=np.arange(8), held_ids=np.arange(8, 12), groups=[np.arange(4), np.arange(4, 8)],
        factors=np.r_[np.ones(8), np.zeros(4)])
    settings = dict(steps=4, batch_size=6, learning_rate=.001, weight_decay=.0001,
        minimum_lr_ratio=.01, checkpoint_every=2, heartbeat_every=1, gradient_clip=5.)
    def run(name, resume=False, stop_at=None):
        torch.manual_seed(17); model = CameraFactorForecaster(ARCH, 1, arm)
        result = fit(model, data, fold, seed=17, settings=settings, identity={'operator_arm': arm},
            directory=tmp_path/name, arm='camera_frame', resume=resume, stop_at=stop_at)
        assert result['held_rows_sampled'] == 0
        return torch.load(tmp_path/name/'checkpoint.pt', weights_only=False)
    direct = run('direct'); run('resume', stop_at=2)
    from scripts.run_m3w_moving_source_forecast import same_scientific_checkpoint
    same_scientific_checkpoint(direct, run('resume', resume=True))


def test_invalid_arm_or_operator_shape_rejected():
    with pytest.raises(ValueError): CameraFactorForecaster(ARCH, 1, 'outcome_selected_best')
    x = inputs(); x['predicted_camera_maps'] = x['predicted_camera_maps'][:, :11]
    with pytest.raises(ValueError): operator_inputs(x, 'past_only')
