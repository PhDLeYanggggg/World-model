import numpy as np

from src.world_model.m3w_vehicle_angular_pair import candidates
from src.world_model.m3w_vehicle_single_rotation import choose, proxy_gains


def test_interaction_guard_abstains_and_preserves_same_rotating_actor():
    solo = np.array([[.7, .6], [.9, -.3]])
    pair = np.array([[-.2, .3], [-.9, -.9]])
    x = choose(solo, pair)
    assert x == dict(independent=(1, 1), interaction=(0, 2), independent_at_interaction=(0, 1))
    assert choose(solo, -np.abs(pair))['interaction'] is None
    assert choose(-solo.clip(min=0), pair)['independent'] is None


def test_proxy_has_no_positive_gain_if_every_surrogate_is_the_incumbent():
    rng = np.random.default_rng(7)
    h = rng.normal(size=(4, 8, 2))
    cv = rng.normal(size=(4, 12, 2))
    anchor = cv+rng.normal(size=cv.shape)
    rows = np.array([0, 2])
    a, b = proxy_gains(candidates(cv[rows], anchor[rows]), anchor, np.stack([anchor]*3), h, rows, np.ones(4))
    assert (a <= 0.).all() and (b <= 0.).all()
    assert all(z is None for z in choose(a, b).values())
