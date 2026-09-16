import pytest
import torch

from test_m3w_supervised_intervention import datasets, training_config
from src.world_model.m3w_supervised_intervention import forecast_smooth_l1, train_forecaster


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_robust_coordinate_loss_bounds_large_residual_gradient():
    prediction = torch.tensor([[[1000., 0.]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = forecast_smooth_l1(prediction, target, torch.ones(1, 1, dtype=torch.bool))
    loss.backward()
    assert loss.item() == pytest.approx(499.75)
    assert prediction.grad.abs().max().item() == .5


def test_robust_loss_ignores_only_declared_invalid_labels():
    prediction = torch.tensor([[[1., -1.], [float('nan'), float('nan')]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    assert forecast_smooth_l1(prediction, target, mask).item() == .5
    with pytest.raises(ValueError):
        forecast_smooth_l1(prediction, target, torch.zeros_like(mask))


def test_robust_fit_can_resume_and_cannot_change_objective(tmp_path):
    _, train, _ = datasets(tmp_path)
    architecture, settings = training_config()
    settings['objective'] = 'smooth_l1'
    partial = train_forecaster(train, architecture=architecture, settings=settings,
                               output_dir=tmp_path / 'robust', stop_after=4)
    assert not partial['training_complete']
    with pytest.raises(ValueError, match='identity changed'):
        train_forecaster(train, architecture=architecture, settings={**settings, 'objective': 'mse'},
                         output_dir=tmp_path / 'robust', resume=True)
    resumed = train_forecaster(train, architecture=architecture, settings=settings,
                               output_dir=tmp_path / 'robust', resume=True)
    continuous = train_forecaster(train, architecture=architecture, settings=settings,
                                  output_dir=tmp_path / 'continuous')
    assert resumed['losses'] == continuous['losses']
    a = torch.load(resumed['checkpoint'], weights_only=True)['model']
    b = torch.load(continuous['checkpoint'], weights_only=True)['model']
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
