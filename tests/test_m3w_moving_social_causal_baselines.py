import numpy as np
import pytest

from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model.m3w_moving_social_causal_baselines import CANDIDATES, accelerations, rollout, fitting_choice


def test_constant_common_velocity_is_exact_causal_cv():
    t = np.arange(8.)
    h = np.stack([np.column_stack((2*t+i, -t+3*i)) for i in range(4)])
    g, _, context = encode_scene(h, np.ones((4, 8), bool), np.arange(4))
    a = accelerations(h, g, context)
    expected = (h[:, -1]-h[:, -2])[:, None]*np.arange(1, 13)[None, :, None]
    for candidate in CANDIDATES[6:]:
        np.testing.assert_allclose(rollout(h, a[candidate['family']], candidate), expected, atol=1e-5)


def test_common_constant_acceleration_recovers_exact_unattenuated_motion():
    t = np.arange(8.)
    h = np.stack([np.column_stack((.5*t*t+i, 2*t+3*i)) for i in range(3)])
    g, _, context = encode_scene(h, np.ones((3, 8), bool), np.arange(3))
    a = accelerations(h, g, context)
    candidate = next(c for c in CANDIDATES if c['family']=='nearest8' and c['alpha']==1 and c['decay']==1)
    expected = np.column_stack((.5*((7+np.arange(1,13))**2-49), 2*np.arange(1,13)))
    np.testing.assert_allclose(rollout(h, a['nearest8'], candidate), np.broadcast_to(expected, (3,12,2)))
    with pytest.raises(TypeError):
        accelerations(h, g, context, future_endpoint=np.zeros((3,2)))


def test_fitting_selection_has_no_held_cost_argument_and_keeps_common_support():
    costs = np.ones((4,len(CANDIDATES)))
    costs[:,6] = .5
    chosen, _ = fitting_choice(costs, np.array(['P0']*4), list(range(15)))
    assert chosen == 6
    costs[0,2] = np.nan
    with pytest.raises(ValueError, match='Common known'):
        fitting_choice(costs, np.array(['P0']*4), list(range(15)))
