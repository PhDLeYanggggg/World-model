import numpy as np
import pytest

from src.world_model.m3w_kinematic_forecast import KinematicForecaster
from src.world_model.m3w_native_prefix_inference import prepare_prefix, predict_relative


def prefix():
    xy = np.arange(8, dtype=float)[None, :, None]*np.array([[[2., 1.]], [[4., 0.]], [[-2., 1.]]])
    mask = np.ones((3, 8), bool)
    mask[1, :3] = False
    xy[~mask] = np.nan
    return dict(positions=xy, history_mask=mask, history_complete=mask.all(1),
                observation_frames=np.arange(4, 12), agent_ids=np.array([23, 7, 11]), query_frame=11)


def test_complete_targets_share_relative_causal_forecasts_and_preserve_order():
    raw = prefix()
    z = prepare_prefix(raw)
    np.testing.assert_array_equal(z.agent_id, [23, 11])
    assert z.geometry.shape == (2, 476)
    assert np.all(z.geometry[:, 230:238] == 1)
    assert np.all(z.geometry[:, 238:294] == 0)
    cv = z.baselines['constant_velocity_causal_fd']
    np.testing.assert_array_equal(cv[:, -1], [[24., 12.], [-24., 12.]])
    np.testing.assert_array_equal(cv, z.baselines['history_ols4_velocity'])
    assert predict_relative(KinematicForecaster(False), z.geometry, batch_size=2).shape == cv.shape


def test_future_fields_and_empty_target_handling():
    raw = prefix()
    with pytest.raises(ValueError, match='observed annotation'):
        prepare_prefix({**raw, 'future_xy': np.empty((3, 12, 2))})
    raw['history_mask'][:, 0] = False
    raw['history_complete'][:] = False
    z = prepare_prefix(raw)
    assert z.geometry.shape == (0, 476)
    assert predict_relative(None, z.geometry, batch_size=2).shape == (0, 12, 2)
