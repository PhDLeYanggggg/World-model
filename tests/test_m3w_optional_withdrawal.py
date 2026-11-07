import numpy as np

from scripts import run_m3w_optional_withdrawal as run
from src.world_model import m3w_optional_withdrawal as m


def test_teacher_marks_pair_reachable_only_after_insertion():
    base = np.array([10., 0., 1., 0., 0., -1.])
    add = np.array([[10., 2., 0., 0., -.1, .1]])
    drop = base[None]
    args = add, drop, np.array([1.]), np.array([2])
    for drop_q, want in ((1, False), (2, True), (3, True)):
        reached, _, label = m.teach(*args, np.array([drop_q]), np.array([2]), np.array([4]),
                                     base, 2., np.random.default_rng(7))
        assert bool(reached[0]) == want
        assert bool(label.any()) == want


def test_routes_keep_insertion_identical_and_do_not_withdraw_earlier():
    ins = (np.array([0, 1]), np.array([1, 0]), np.array([1, 3]), np.array([[.8], [.9]]))
    drops = np.array([2, 3, 4]), np.array([0, 1, 4]), np.array([[1.], [.7], [.95]])
    for future_score in (.95, .1):
        changed = (drops[0], drops[1], np.array([[1.], [.7], [future_score]]))
        routes = run.routes(ins, changed, lambda x: x[:, 0], lambda x: x[:, 1])
        assert routes[(.5, None)] == (0, 1, -1)
        assert routes[(.5, .5)] == (0, 1, 3)


def test_minibatch_positive_quota_is_deterministic():
    y = np.r_[True, np.zeros(999, dtype=bool)]
    x = m.plan(y, np.ones(len(y)), 9)
    np.testing.assert_array_equal(x, m.plan(y, np.ones(len(y)), 9))
    assert (x[:, :m.BATCH//4] == 0).all()


def test_unlabelled_withdrawal_is_not_a_free_gain():
    a = np.array([[[10., 2., 0., 0., -.1, .1]],
                  [[0., 0., .6, .6, 0., -.6]]])
    original = a[1, 0]
    risk, gain = run.observe((0, 0, 1), a, np.array([True, False]), np.array([-1, 0]),
                              original, 2., np.array([2, 3]))
    assert risk['supported']
    np.testing.assert_allclose(gain, .05-.2)
