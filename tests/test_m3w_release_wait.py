import numpy as np

from src.evaluation.m3w_streaming_menu_capacity import respects_tokens
from src.world_model.m3w_release_wait import schedule


def test_wait_cannot_consume_reserved_future_anchor_credit():
    t = np.array([4, 9, 10, 11, 14])
    rows = np.arange(len(t))
    slots = np.array([0, 1, 1, 1, 2])
    release = np.array([4, 9, 14])
    eligible = np.ones(15, dtype=bool)
    original = np.array([0, 1, 4])
    cap = np.full(15, 2)
    score = np.array([1., 1., 1., 2., 1.])
    result = schedule(t, rows, slots, score, release, original, eligible, cap, np.array([False, True, False]))
    np.testing.assert_array_equal(result, [0, 2, 4])
    assert respects_tokens(np.bincount(t[result], minlength=15), eligible)


def test_declined_anchor_expires_without_backdating():
    t = np.array([4, 7, 9])
    release, slots = np.array([4, 9]), np.array([0, 0, 1])
    got = schedule(t, np.arange(3), slots, np.ones(3), release, np.array([0, 2]),
                   np.ones(10, dtype=bool), np.ones(10, dtype=int), np.array([True, False]))
    np.testing.assert_array_equal(got, [2])
