import numpy as np

from src.world_model.m3w_wait_peer_value import causal_pick, opportunity_features, peer_features


def test_peer_control_stays_in_the_same_observation_query():
    rng = np.random.default_rng(91)
    h = rng.normal(size=(6, 8, 2)).cumsum(1)
    q = np.repeat(np.arange(2), 3)
    aligned, wrong = peer_features(h, q, [0, 1, 3, 4])
    assert aligned.shape == wrong.shape == (4, 34)
    np.testing.assert_array_equal(wrong[[0, 2]], aligned[[1, 3]])
    changed = h.copy()
    changed[3:] += 500.
    changed[3:] *= -4.
    z, w = peer_features(changed, q, [0, 1, 3, 4])
    np.testing.assert_array_equal(z[:2], aligned[:2])
    np.testing.assert_array_equal(w[:2], wrong[:2])


def test_causal_choice_and_history_ignore_later_queries():
    t = np.array([4, 4, 5, 7])
    value = np.ones(len(t))
    prob = np.array([.3, .4, .7, .9])
    assert causal_pick(t, prob, value, value) == 1
    prob[2:] = 0.
    assert causal_pick(t, prob, value, value) == 1
    q = np.repeat(np.arange(8), 2)
    radii = np.ones(len(q))
    cap = np.ones(8)
    base = opportunity_features(q, radii > 0, radii, cap, t)
    radii[q > 5] = 1000.
    other = opportunity_features(q, radii > 0, radii, cap, t)
    np.testing.assert_array_equal(base[:3], other[:3])
