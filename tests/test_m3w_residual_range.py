import numpy as np
import pytest
import torch

from src.world_model.m3w_objective_alignment import fit_objective, geometry_prediction
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_residual_range import decode_residual, range_loss, range_prediction, fit_range


def test_decoder_zero_small_large_and_numerical_cap():
    z = torch.tensor([0., 1e-5, 7., 20.], dtype=torch.float64, requires_grad=True)
    y = decode_residual(z, 'sinh', 12.)
    assert y[0] == 0 and abs(y[1].item()-1e-5) < 1e-14
    assert y[2] > 500 and y[3] == torch.sinh(torch.tensor(12., dtype=torch.float64))
    y.sum().backward()
    assert z.grad[0] == 1 and z.grad[3] == 0
    torch.testing.assert_close(torch.asinh(y[:3]), z[:3])
    assert torch.equal(decode_residual(z, 'linear', 12.), z)


def test_encoded_loss_is_supervision_not_inference_and_targets_not_clipped():
    p = torch.zeros(2, 12, 2, requires_grad=True)
    b = torch.zeros_like(p)
    y = torch.full_like(p, 1e6)
    loss, detail = range_loss(p, y, b, 'asinh')
    assert torch.isfinite(loss) and loss > 10
    loss.backward()
    assert p.grad.abs().min() > .01
    assert detail['ADE'] > 1e6
    with pytest.raises(ValueError):
        range_loss(p, y, b, 'unknown')


def fixture():
    torch.manual_seed(17)
    x, observed = torch.randn(20, 16), torch.ones(20, 8)
    base, target = torch.zeros(20, 12, 2), torch.randn(20, 12, 2)
    return lambda ids: (x[ids], observed[ids], base[ids], target[ids])


def test_zero_initialization_and_linear_prediction_matches_control():
    model = OfflineVisualForecast(16)
    x, observed, base, target = fixture()(torch.arange(20))
    for arm in ('linear_log', 'sinh_log', 'linear_asinh', 'sinh_asinh'):
        np.testing.assert_array_equal(range_prediction(model, x, observed, base, arm, 12.).detach(), base)
    torch.nn.init.normal_(model.output.weight)
    assert torch.equal(range_prediction(model, x, observed, base, 'linear_log', 12.),
                       geometry_prediction(model, x, observed, base))
    with pytest.raises(ValueError):
        range_prediction(model, x, observed, base, 'unknown', 12.)


def test_control_training_matches_old_and_resume_is_exact(tmp_path):
    torch.set_num_threads(4)
    cfg = dict(updates=80, batch_size=8, learning_rate=.0003, weight_decay=.0001, checkpoint_every=40)
    batch, ids = fixture(), torch.arange(20)
    identity = {'fixture': True}
    torch.manual_seed(17)
    old = OfflineVisualForecast(16)
    fit_objective(old, batch, ids, np.repeat([0, 1], 10), arm='row_log', config=cfg,
                  seed=17, identity=identity, checkpoint=tmp_path/'old.pt', heartbeat=lambda v: None)
    for arm in ('linear_log', 'sinh_asinh'):
        torch.manual_seed(17)
        full = OfflineVisualForecast(16)
        fit_range(full, batch, ids, arm=arm, cap=12., config=cfg, seed=17, identity=identity,
                  checkpoint=tmp_path/(arm+'_full.pt'), heartbeat=lambda v: None)
        torch.manual_seed(17)
        split = OfflineVisualForecast(16)
        cp = tmp_path/(arm+'_split.pt')
        fit_range(split, batch, ids, arm=arm, cap=12., config=cfg, seed=17, identity=identity,
                  checkpoint=cp, heartbeat=lambda v: None, stop_at=40)
        result = fit_range(split, batch, ids, arm=arm, cap=12., config=cfg, seed=17, identity=identity,
                           checkpoint=cp, heartbeat=lambda v: None)
        assert result['new_updates_this_invocation'] == 40
        for k in full.state_dict():
            assert torch.equal(full.state_dict()[k], split.state_dict()[k])
            if arm == 'linear_log':
                assert torch.equal(full.state_dict()[k], old.state_dict()[k])
        with pytest.raises(ValueError, match='identity'):
            fit_range(split, batch, ids, arm=arm, cap=11., config=cfg, seed=17, identity=identity,
                      checkpoint=cp, heartbeat=lambda v: None)
