import numpy as np
import pytest

from src.world_model.m3w_opendd_feedback_recipient import traces


def run(starts, targets, rec=("a", "a", "a")):
    ids = np.tile([1, 2], 3)
    h = np.zeros((6, 8, 2))
    h[:, :, 0] = np.arange(8)
    reads = []

    def observe(at, last, now):
        assert last < now
        reads.extend(at)
        return targets[at]

    feat, n = traces(h, ids, np.arange(0, 7, 2), np.array(rec), np.array(starts), 1, observe)
    return feat, n, reads


def test_current_targets_do_not_enter_same_query_and_shuffle_preserves_support():
    y = np.ones((6, 12, 2))
    y[0] *= 3.
    x, n, reads = run([0, 20, 40], y)
    y[2:4] *= 5.
    z, _, _ = run([0, 20, 40], y)
    assert reads == [0, 1, 2, 3]
    assert n == 4
    np.testing.assert_array_equal(x["aligned"][:4], z["aligned"][:4])
    assert not np.array_equal(x["aligned"][4:], z["aligned"][4:])
    np.testing.assert_array_equal(x["aligned"][:, :3], x["wrong_agent"][:, :3])
    np.testing.assert_array_equal(x["aligned"][2, 3:], x["wrong_agent"][3, 3:])
    assert not x["availability"][:, 3:].any()


def test_never_reads_before_maturity_or_across_recordings():
    y = np.ones((6, 12, 2))
    with pytest.raises(ValueError, match="overlap"):
        run([0, 19, 40], y)
    f, _, _ = run([0, 20, 40], y, ("a", "a", "b"))
    assert not f["aligned"][4:].any()


def test_unobserved_past_labels_are_not_invented():
    y = np.ones((6, 12, 2))
    y[0] = np.nan
    f, n, _ = run([0, 20, 40], y)
    np.testing.assert_array_equal(f["aligned"][2], np.zeros(7))
    assert n == 2
