import numpy as np

from scripts.run_m3w_ledger_wait import contrast
from src.world_model.m3w_ledger_wait import local_wait, walk
from src.world_model.m3w_release_wait import schedule


def test_local_opportunity_can_conflict_with_a_later_untouched_edit():
    q = np.array([4, 6, 6, 6])
    rows = np.array([1, 2, 2, 3])
    slot = np.array([0, 0, 1, 1])
    proxy = np.array([.2, .9, .1, .5])
    release, old = np.array([4, 6]), np.array([0, 2])
    cap, eligible = np.full(7, 2), np.array([1, 1, 1, 1, 1, 0, 1])
    # Start with banked credit for two offers.
    q, release = q+5, release+5
    cap, eligible = np.r_[np.full(5, 2), cap], np.r_[np.ones(5, dtype=int), eligible]
    np.testing.assert_array_equal(local_wait(q, slot, proxy, release, old), [1, -1])
    flags = np.array([True, False])
    args = q, rows, slot, proxy, release, old, eligible, cap
    got, state, chosen = walk(*args, lambda a, _: flags[a])
    np.testing.assert_array_equal(got, [2])
    np.testing.assert_array_equal(chosen, flags)
    np.testing.assert_array_equal(got, schedule(*args, flags))
    earlier, earlier_state, _ = walk(*args, lambda a, _: np.array([False, True])[a])
    assert earlier[0] == 0
    np.testing.assert_array_equal(earlier_state[0], state[0])


def test_contrast_debits_missing_steps_and_includes_other_edges():
    cv = np.zeros((3, 2, 2))
    y = np.zeros_like(cv)
    mask = np.array([[True, False], [True, True], [True, True]])
    y[1, :, 0] = 1.
    paths = np.zeros_like(cv)
    paths[:, :, 0] = 1.
    risk = np.arange(9).reshape(3, 3)
    delta = contrast(cv, y, mask, np.arange(3), paths, risk, [0, 1], [2])
    np.testing.assert_allclose(delta, [-2., 3., 2., 1.])
