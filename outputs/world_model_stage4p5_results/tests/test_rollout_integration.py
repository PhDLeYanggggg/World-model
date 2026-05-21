import numpy as np

from src.data.synthetic_physical_crowd import STATE_COLUMNS
from src.models.baselines import constant_acceleration_rollout, constant_velocity_rollout
from src.physics.scene_geometry import SceneSpec


def make_state(x=0.0, y=0.0, vx=1.0, vy=0.0, ax=0.0, ay=0.0):
    state = np.zeros((1, len(STATE_COLUMNS)), dtype=np.float32)
    state[0, 0:6] = [x, y, vx, vy, ax, ay]
    state[0, 7] = 0.3
    state[0, 8] = 99.0
    state[0, 9] = 99.0
    return state


def test_constant_velocity_straight_track_zero_error():
    scene = SceneSpec("unit", 100, 100, [], {}, [], "test", False, False, False)
    dt = 0.1
    history = np.stack([make_state(x=t * dt) for t in range(6)], axis=0)
    pred = constant_velocity_rollout(history, scene, 10, dt, 99, 99, False, False)
    expected_x = history[-1, 0, 0] + np.arange(11) * dt
    np.testing.assert_allclose(pred[:, 0, 0], expected_x, atol=1e-6)


def test_constant_acceleration_beats_cv_on_accelerating_track():
    scene = SceneSpec("unit", 100, 100, [], {}, [], "test", False, False, False)
    dt = 0.1
    a = 1.0
    states = []
    for t in range(6):
        time = t * dt
        states.append(make_state(x=0.5 * a * time * time, vx=a * time, ax=a))
    history = np.stack(states, axis=0)
    truth = 0.5 * a * ((5 + 10) * dt) ** 2
    cv = constant_velocity_rollout(history, scene, 10, dt, 99, 99, False, False)[-1, 0, 0]
    ca = constant_acceleration_rollout(history, scene, 10, dt, 99, 99, False, False)[-1, 0, 0]
    assert abs(ca - truth) < abs(cv - truth)
