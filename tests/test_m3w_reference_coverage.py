import numpy as np
import pytest
import torch

from src.world_model.m3w_baseline_relative_forecaster import past_motion_budget, radial_squash
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_european_source_forecast import baseline_torch, fit_design
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_reference_coverage import (
    ReferenceCompleteAgentTrackForecaster, reference_motion_budget,
)
from src.world_model.m3w_reference_harm_forecast import fit_trial, assert_training_equivalent


def geometry(n=12):
    g = np.zeros((n, 476), np.float32)
    xy = np.zeros((8, 2), np.float32)
    xy[:, 0] = np.array([-11., -11., -11., -10., -10., -10., -10., 0.])
    g[:, :16] = xy.reshape(-1)
    g[:, 16:24] = np.arange(-7, 1)/12
    return g


def test_reference_coverage_repairs_realistic_causal_counterexample():
    inputs = pack_geometry(geometry(2))
    inputs['baseline'] = baseline_torch(inputs['history'], 4)
    old, new = past_motion_budget(inputs), reference_motion_budget(inputs)
    cv = baseline_torch(inputs['history'], 1)
    distance = (cv-inputs['baseline']).norm(dim=-1)
    assert (distance > old).any()
    assert (distance < new).all()
    assert (new >= old).all()
    for index in (1, 3):
        difference = baseline_torch(inputs['history'], index)-inputs['baseline']
        fraction = difference/new[..., None]
        inverse = fraction/torch.sqrt(1-fraction.square().sum(-1, keepdim=True))
        reconstructed = inputs['baseline']+new[..., None]*radial_squash(inverse)
        torch.testing.assert_close(reconstructed, baseline_torch(inputs['history'], index),
                                   rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(reference_motion_budget(inputs, enabled=False), old, rtol=0, atol=0)


def test_stationary_missing_requests_and_extra_future_fields():
    inputs = pack_geometry(geometry(1))
    inputs['history'][..., :2] = 0.
    inputs['baseline'] = baseline_torch(inputs['history'], 1)
    assert torch.equal(reference_motion_budget(inputs), torch.zeros((1, 12)))
    inputs['request_mask'][:, -1] = False
    assert reference_motion_budget(inputs)[0, -1] == 0
    with pytest.raises(ValueError):
        reference_motion_budget(dict(inputs, future_endpoint=torch.zeros(1, 2)))
    for margin in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            reference_motion_budget(inputs, margin=margin)


def test_initialization_disabled_forward_and_parameter_budget_are_exact():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    old = DimensionlessAgentTrackSourceForecaster(4, width=16, heads=2, layers=2)
    torch.manual_seed(17)
    neutral = ReferenceCompleteAgentTrackForecaster(4, width=16, heads=2, layers=2, enabled=False)
    assert list(old.state_dict()) == list(neutral.state_dict())
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, neutral.state_dict()[key], rtol=0, atol=0)
    inputs = pack_geometry(geometry(3))
    old.eval(); neutral.eval()
    torch.testing.assert_close(old(inputs), neutral(inputs), rtol=0, atol=0)
    with pytest.raises(ValueError):
        neutral(dict(inputs, future_goal=torch.ones(3, 2)))


def test_fixed_training_resume_and_coverage_off_replay_exact(tmp_path):
    torch.set_num_threads(4)
    g = geometry()
    data = dict(geometry=g, target=np.tile(np.arange(1, 13).repeat(2), (12, 1)).reshape(-1, 12, 2),
                valid=np.ones((12, 12), bool), sites=np.repeat(['a', 'b', 'c'], 4))
    fold = fit_design(data, ['a', 'b'], np.tile([3., 2., 3., 2., 1., 2.], (12, 1)))
    settings = dict(steps=6, batch_size=4, learning_rate=.001, minimum_lr_ratio=.01,
                    weight_decay=.0001, gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    for arm in ('control', 'neutral', 'coverage', 'resumed'):
        torch.manual_seed(17)
        cls = DimensionlessAgentTrackSourceForecaster if arm == 'control' else ReferenceCompleteAgentTrackForecaster
        model = cls(4, width=16, heads=2, layers=2,
                    **({} if arm == 'control' else dict(enabled=arm != 'neutral')))
        kw = dict(seed=17, settings=settings, identity=dict(fixture=True),
                    directory=tmp_path/arm, heartbeat=lambda **kw:None)
        fit_trial(model, data, fold, **kw, stop_at=3 if arm == 'resumed' else None)
        if arm == 'resumed':
            fit_trial(ReferenceCompleteAgentTrackForecaster(4, width=16, heads=2, layers=2),
                      data, fold, **kw, resume=True)
    load = lambda arm:torch.load(tmp_path/arm/'checkpoint.pt', weights_only=False)
    assert assert_training_equivalent(load('control'), load('neutral'))
    assert assert_training_equivalent(load('coverage'), load('resumed'))
