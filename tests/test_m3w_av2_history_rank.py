import numpy as np

from scripts import run_m3w_av2_history_rank as run
from scripts import audit_m3w_av2_history_rank as audit
from src.world_model import m3w_av2_history_replay as history


def test_prefix_alignment_rotates_heading_and_scales_speed():
    delta = np.array([[[1., 0.], [2., 1.]]])
    got = history.align(delta, np.array([[1., 0.]]), np.array([[0., 2.]]))
    np.testing.assert_allclose(got, [[[0., 2.], [-2., 4.]]])
    np.testing.assert_array_equal(history.align(delta, np.zeros((1, 2)), np.ones((1, 2))), delta)


def test_replay_is_rigid_motion_invariant_and_recipient_order_invariant():
    rng = np.random.default_rng(21)
    h = rng.normal(size=(3, 8, 2)).cumsum(1)
    cv = history.motion.rollout(h, 12)[0]
    p = np.repeat(cv[None], 2, axis=0)
    rows = np.array([[0, 1], [0, 2]])
    p[np.arange(2)[:, None], rows] += rng.normal(size=(2, 2, 12, 2))
    expected = history.features(h, cv, p, rows, .3)
    r = np.array([[0., 1.], [-1., 0.]])
    other = history.features(h@r+np.array([8., 13.]), cv@r, p@r, rows[:, ::-1], .3)
    assert expected.shape == (2, history.WIDTH)
    np.testing.assert_allclose(expected, other, atol=1e-6)


def test_rank_loss_rewards_the_qualified_candidate_over_the_negative():
    t = run.torch
    p, n = t.tensor([.7, .8]), t.tensor([.3, .4])
    assert run.rank_loss(p, n) < run.rank_loss(n, p)
    assert t.isfinite(run.rank_loss(t.zeros(1), t.ones(1)))


def test_geometry_arm_masks_only_the_new_history_channels():
    t = run.torch
    x, h = t.randn(3, 5), t.randn(3, 2)
    g = run.with_history(x, h, "geometry")
    d = run.with_history(x, h, "history")
    t.testing.assert_close(g[:, :5], d[:, :5])
    t.testing.assert_close(g[:, 5:], t.zeros_like(h))
    t.testing.assert_close(d[:, 5:], h)


def test_direct_trajectory_charges_missing_disagreement_on_same_denominator():
    cv = np.array([[[2., 0.], [2., 0.]]])
    y = np.zeros_like(cv)
    v = np.array([[True, False]])
    after = np.array([[[1., 0.], [4., 0.]]])
    np.testing.assert_allclose(audit.direct_trajectory(after, cv, cv, y, v), [50., 100., -50.])
