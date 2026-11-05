import numpy as np

from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_kinematic_risk import motion_basis


def test_basis_matches_real_geometry_to_cost_feature_builder():
    history = np.zeros((2, 8, 2)); history[0, :, 0] = np.arange(-7, 1); history[1, :, 0] = 2
    geometry, ids, _ = encode_scene(history, np.ones((2, 8), bool), np.array([10, 20]))
    np.testing.assert_array_equal(ids, [0, 1])
    floor = geometry[:, 332:356].reshape(2, 12, 2)
    features, _ = matched_features(geometry, floor, floor, floor*1.2, np.zeros(2, bool))
    x = np.broadcast_to(np.column_stack((features, np.zeros((2, 48))))[:, None], (2, 7, 428)).copy()
    basis = motion_basis(x)
    np.testing.assert_allclose(basis[0, :, 14:21], 1/7, rtol=1e-6)
    np.testing.assert_allclose(basis[0, :, 42:50], np.broadcast_to(
        [1/8, 2/7, 2/7, 1/7, 1/7, 2/12, 0, 1], (7, 8)), atol=1e-7)
    # A learned rollout diagnostic is not a ground-truth future endpoint.
    np.testing.assert_allclose(basis[0, :, 50], 12/7, rtol=1e-6)
    np.testing.assert_allclose(basis[0, :, 57], 1.2*12/7, rtol=1e-6)
