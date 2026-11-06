"""Synthetic representation counterexample, not a real forecasting score."""
import numpy as np
import torch

from src.world_model.m3w_camera_factor import CameraFactorForecaster
from src.world_model.m3w_camera_frame import pack_inputs, query_maps
from tests.test_m3w_visual_forecast import ARCH


def request(camera_velocity):
    camera_velocity = np.asarray(camera_velocity, np.float64)
    world_velocity = np.array([1., 0.])
    history = np.arange(-7, 1)[:, None]*(world_velocity+camera_velocity)
    g = np.zeros((1, 476), np.float32)
    g[0, :16] = history.reshape(-1); g[0, 16:24] = np.arange(-7, 1)/12
    # No visible neighbors in this lawful counterexample; eligibility uses past only.
    affine = np.eye(3); affine[:2, 2] = camera_velocity
    maps = query_maps({f: dict(supported=True, native_affine=affine.tolist()) for f in range(1, 8)}, 7)
    batch = pack_inputs(g, np.zeros((1, 2)), maps['past'][None], maps['prediction'][None], 'camera_frame')
    target = np.arange(1, 13)[None, :, None]*(world_velocity+camera_velocity)
    return batch, torch.from_numpy(target).float()


def test_past_only_can_erase_distinct_causal_camera_motion_needed_by_native_labels():
    stationary_view, labels_a = request([0., 0.])
    moving_view, labels_b = request([2., -1.])
    assert not torch.equal(stationary_view['history'], moving_view['history'])
    assert not torch.equal(labels_a, labels_b)
    torch.manual_seed(17); model = CameraFactorForecaster(ARCH, 1, 'past_only'); model.eval()
    with torch.no_grad():
        prediction_a, prediction_b = model(stationary_view), model(moving_view)
    torch.testing.assert_close(prediction_a, prediction_b, atol=0, rtol=0)


def test_both_operator_arm_retains_camera_information_at_the_output():
    stationary_view, _ = request([0., 0.]); moving_view, _ = request([2., -1.])
    torch.manual_seed(17); model = CameraFactorForecaster(ARCH, 1, 'both'); model.eval()
    with torch.no_grad():
        difference = model(moving_view)-model(stationary_view)
    expected = torch.arange(1, 13)[None, :, None]*torch.tensor([2., -1.])
    torch.testing.assert_close(difference, expected, atol=1e-6, rtol=1e-6)
