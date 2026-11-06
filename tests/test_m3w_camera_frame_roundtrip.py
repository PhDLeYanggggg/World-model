"""Additional unit-only coordinate checks; no experiment definition changes."""
import numpy as np
import torch

from src.world_model.m3w_camera_frame import CameraFrameForecaster, query_maps, pack_inputs
from src.world_model.m3w_european_source_forecast import SourceForecaster
from tests.test_m3w_camera_frame import ARCH, flows, geometry


def test_network_wrapper_restores_native_relative_origin_and_camera_motion(monkeypatch):
    angle = .06
    a = np.array([[np.cos(angle), -np.sin(angle), 2.],
        [np.sin(angle), np.cos(angle), -1.], [0., 0., 1.]])
    operators = query_maps(flows(a), 7)
    origins = np.array([[50., 30.], [100., 60.]])
    history = np.stack([np.array([(np.linalg.matrix_power(a, t)@np.r_[o, 1.])[:2]
        for t in range(-7, 1)]) for o in origins])
    expected = np.stack([np.array([(np.linalg.matrix_power(a, t)@np.r_[o, 1.])[:2]-o
        for t in range(1, 13)]) for o in origins])
    g = geometry(2); g[:, :16] = (history-origins[:, None]).reshape(2, 16)

    def canonical_cv(self, inputs):
        h = inputs['history'][..., :2]
        t = torch.arange(1, 13, dtype=h.dtype)[None, :, None]
        return h[:, -1, None]+t*(h[:, -1]-h[:, -2])[:, None]

    monkeypatch.setattr(SourceForecaster, 'forward', canonical_cv)
    model = CameraFrameForecaster(ARCH, 1, 'camera_frame')
    output = model(pack_inputs(g, origins, np.broadcast_to(operators['past'], (2, 8, 3, 3)),
        np.broadcast_to(operators['prediction'], (2, 12, 3, 3)), 'camera_frame')).numpy()
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=3e-4)
