import numpy as np
import pytest
import torch

from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_objective_alignment import (
    fit_objective, geometry_prediction, objective_loss, sampling_weights, supported_standardization,
)


def test_scene_sampling_equals_scene_risk_not_window_risk():
    folds = np.array([0, 1, 1, 1])
    weights = sampling_weights(folds, 'scene').numpy()
    np.testing.assert_allclose(weights, [.5, 1/6, 1/6, 1/6])
    errors = np.array([12., 1., 2., 3.])
    assert (weights * errors).sum() == pytest.approx(7., abs=1e-12)
    np.testing.assert_allclose(sampling_weights(folds, 'row'), np.ones(4)/4)


def test_training_constants_and_query_statistics_do_not_leak():
    train = np.array([[120., 1.], [120., 3.]], np.float32)
    query = np.array([[72., 2.]], np.float32)
    x, normalizer = supported_standardization(train, query)
    y, _ = supported_standardization(train, np.vstack((query, [999., 999.])))
    np.testing.assert_array_equal(x, y[:1])
    np.testing.assert_array_equal(x, [[0., 0.]])
    assert normalizer['constant'].tolist() == [True, False]


def test_geometry_fast_path_exactly_matches_frozen_model():
    torch.manual_seed(2)
    model = OfflineVisualForecast(5)
    with torch.no_grad():
        model.output.weight.normal_()
    x = torch.randn(3, 5)
    rgb, coverage = torch.rand(3, 8, 3, 32, 32), torch.rand(3, 8, 1, 32, 32)
    baseline = torch.randn(3, 12, 2)
    assert torch.equal(geometry_prediction(model, x, coverage.mean((2, 3, 4)), baseline),
                       model(x, rgb, coverage, baseline, 'geometry'))


def test_losses_and_gradients_show_log_downweighting_and_harm_penalty():
    target = torch.zeros(2, 12, 2)
    prediction = torch.zeros_like(target)
    prediction[0, :, 0] = 1.
    prediction[1, :, 0] = 99.
    prediction.requires_grad_()
    log, _ = objective_loss(prediction, target, target, 'log')
    linear, _ = objective_loss(prediction, target, target, 'ade')
    penalized, _ = objective_loss(prediction, target, target, 'ade_harm')
    assert linear.item() == 50.
    assert penalized.item() == 100.
    log_grad, = torch.autograd.grad(log, prediction, retain_graph=True)
    ade_grad, = torch.autograd.grad(linear, prediction)
    assert torch.allclose(log_grad[1, :, 0], ade_grad[1, :, 0] / 100)
    zero = target.clone().requires_grad_()
    loss, _ = objective_loss(zero, target, target, 'ade_harm')
    loss.backward()
    assert torch.isfinite(zero.grad).all()
    with pytest.raises(ValueError):
        objective_loss(prediction, target, target, 'test_selected')


def test_paired_statistics_use_ratio_of_scene_means_not_percent_average():
    from scripts.run_m3w_objective_alignment import paired_interval
    baseline = np.tile([1., 10., 100.], (3, 1))
    prediction = np.tile([2., 9., 90.], (3, 1))
    result = paired_interval(prediction, baseline, 2000)
    assert result['gain_percent'] == pytest.approx(100 * (1 - 101/111))
    assert result['not_independent_confirmation'] is True
    assert result['scene_clusters'] == 3
    with pytest.raises(ValueError):
        paired_interval(prediction[:1], baseline[:1], 2000)


@pytest.mark.parametrize('arm', ['row_log', 'scene_ade_harm'])
def test_exact_resume_and_training_rows_only(tmp_path, arm):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    x, observed = torch.randn(6, 5), torch.rand(6, 8)
    baseline, target = torch.randn(6, 12, 2), torch.randn(6, 12, 2)
    def batch(ids):
        assert torch.all(ids < 5)
        return x[ids], observed[ids], baseline[ids], target[ids]
    config = dict(updates=8, batch_size=4, checkpoint_every=2, learning_rate=.001, weight_decay=.0001)
    identity = {'test': 'exact_resume'}
    def run(path, stop=None):
        torch.manual_seed(17)
        model = OfflineVisualForecast(5)
        result = fit_objective(model, batch, torch.arange(5), [0, 0, 1, 1, 1], arm=arm,
            config=config, seed=17, identity=identity, checkpoint=path, heartbeat=lambda v: None, stop_at=stop)
        return model, result
    full, _ = run(tmp_path/'full.pt')
    _, partial = run(tmp_path/'resume.pt', 3)
    assert not partial['complete']
    resumed, result = run(tmp_path/'resume.pt')
    assert result['new_updates_this_invocation'] == 5
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
    _, cached = run(tmp_path/'resume.pt')
    assert cached['new_updates_this_invocation'] == 0
    with pytest.raises(ValueError):
        fit_objective(full, batch, torch.arange(5), [0, 0, 1, 1, 1], arm=arm, config=config,
            seed=17, identity={'changed': True}, checkpoint=tmp_path/'resume.pt', heartbeat=lambda v: None)
