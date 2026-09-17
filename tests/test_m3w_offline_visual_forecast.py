import numpy as np
import pytest
import torch

from src.world_model.m3w_offline_visual_forecast import (
    ARMS, OfflineVisualForecast, fit_model, forecast_metrics, geometry_features,
)


def inputs():
    torch.manual_seed(4)
    return [torch.randn(5, 9), torch.rand(5, 8, 3, 32, 32),
            torch.ones(5, 8, 1, 32, 32), torch.randn(5, 12, 2), torch.randn(5, 12, 2)]


def test_initial_prediction_is_exact_cv_and_has_finite_gradient():
    values = inputs()
    model = OfflineVisualForecast(9)
    for arm in ARMS:
        assert torch.equal(model(*values[:4], arm), values[3])
    model(*values[:4], 'past_rgb').square().mean().backward()
    assert torch.isfinite(model.output.weight.grad).all()


def test_masked_pixels_and_noncurrent_images_cannot_change_prediction():
    values = inputs()
    model = OfflineVisualForecast(9)
    with torch.no_grad():
        model.output.weight.fill_(.1)
    for arm in ('geometry', 'mask_only', 'current_rgb'):
        original = model(*values[:4], arm)
        altered = [v.clone() for v in values]
        altered[1][:, :7] += 100
        assert torch.equal(original, model(*altered[:4], arm))
    values[2].zero_()
    original = model(*values[:4], 'past_rgb')
    values[1].fill_(float('nan'))
    assert torch.equal(original, model(*values[:4], 'past_rgb'))


def test_geometry_explicit_allowlist_ignores_future_and_source_provenance():
    row = dict(prediction_frame_offsets=np.arange(1, 13), history_xy=np.zeros((8, 2)),
        history_velocity=np.zeros((7, 2)), history_frame_offsets=np.arange(-7, 1),
        neighbor_mask=np.zeros((8, 8), bool), neighbor_xy=np.ones((8, 8, 2)),
        neighbor_frame_offsets=np.ones((8, 8)), causal_features=np.zeros(14),
        baseline_rollouts=np.zeros((7, 12, 2)))
    expected = geometry_features(row)
    row.update(future_endpoint=np.ones(2) * 1e10, future_goal=123, latest_control_frame=1e10)
    assert np.array_equal(expected, geometry_features(row))
    row['neighbor_xy'] *= 1e9
    assert np.array_equal(expected, geometry_features(row))


def test_exact_optimizer_resume(tmp_path):
    torch.set_num_threads(2)
    values = inputs()
    def batch(ids):
        return [v[ids] for v in values]
    config = dict(learning_rate=.001, weight_decay=.001, updates=6, batch_size=3, checkpoint_every=2)
    identity = {'probe': 'resume'}
    def run(name, stop=None):
        torch.manual_seed(17)
        model = OfflineVisualForecast(9)
        fit_model(model, batch, torch.arange(5), arm='past_rgb', config=config, seed=17,
            identity=identity, checkpoint=tmp_path / name, heartbeat=lambda _: None, stop_at=stop)
        return model
    expected = run('full.pt')
    run('split.pt', 2)
    actual = run('split.pt')
    for k, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[k])
    with pytest.raises(ValueError):
        fit_model(actual, batch, torch.arange(5), arm='past_rgb', config=config, seed=17,
            identity={'probe': 'changed'}, checkpoint=tmp_path / 'split.pt', heartbeat=lambda _: None)


def test_metrics_compare_actual_moving_reference_and_no_zero_division_claim():
    target = np.ones((2, 12, 2))
    cv = target * .5
    good = forecast_metrics(target, target, cv, np.ones(2), 1.)
    assert good['improvement_percent'] == 100
    equal = forecast_metrics(cv, target, cv, np.ones(2), 1.)
    assert equal['improvement_percent'] == equal['easy_degradation_percent'] == 0
    perfect = forecast_metrics(target, target, target, np.ones(2), 1.)
    assert perfect['improvement_percent'] is None
