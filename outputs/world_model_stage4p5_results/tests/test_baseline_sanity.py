import numpy as np

from src.data.synthetic_physical_crowd import STATE_COLUMNS
from src.models.baselines import constant_velocity_rollout, damped_velocity_rollout, identity_hand_physics_rollout
from src.physics.scene_geometry import SceneSpec


def history_with_velocity(vx=1.0):
    history = np.zeros((6, 1, len(STATE_COLUMNS)), dtype=np.float32)
    for t in range(6):
        history[t, 0, 0] = t * 0.1 * vx
        history[t, 0, 2] = vx
        history[t, 0, 7] = 0.3
        history[t, 0, 8] = 99.0
        history[t, 0, 9] = 99.0
    return history


def test_identity_hand_physics_matches_constant_velocity_without_forces():
    scene = SceneSpec("no_geometry", 10, 10, [], {}, [], "test", False, False, False)
    history = history_with_velocity(1.0)
    cv = constant_velocity_rollout(history, scene, 20, 0.1, 99, 99, False, False)
    identity = identity_hand_physics_rollout(history, scene, 20, 0.1, 99, 99, False, False)
    np.testing.assert_allclose(identity[:, :, 0:2], cv[:, :, 0:2], atol=1e-6)


def test_damped_velocity_reduces_stopping_drift():
    scene = SceneSpec("no_geometry", 100, 100, [], {}, [], "test", False, False, False)
    history = history_with_velocity(1.0)
    stopped_truth_x = history[-1, 0, 0]
    cv_x = constant_velocity_rollout(history, scene, 50, 0.1, 99, 99, False, False)[-1, 0, 0]
    damped_x = damped_velocity_rollout(history, scene, 50, 0.1, 99, 99, damping=0.8, use_collision_projection=False, use_scene_constraints=False)[-1, 0, 0]
    assert abs(damped_x - stopped_truth_x) < abs(cv_x - stopped_truth_x)
