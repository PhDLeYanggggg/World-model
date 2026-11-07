import numpy as np

from src.world_model import m3w_postsearch_risk as m


def test_matched_count_is_exact_per_recording_and_action_size():
    rec = np.array(['a']*4+['b']*4)
    count = np.tile([1, 1, 2, 2], 2)
    target = np.array([1, 0, 0, 1, 0, 1, 0, 0], bool)
    actual = m.rank_count(np.arange(8), target, rec, count)
    np.testing.assert_array_equal(actual, [0, 1, 0, 1, 0, 1, 0, 0])


def test_postsearch_fit_keeps_outputs_finite_and_bounded():
    x = np.column_stack((np.ones(6), np.arange(6)))
    y = np.column_stack((np.arange(6) > 2, np.arange(6) % 2))
    b = m.fit(x, y, np.repeat(['a', 'b'], 3))
    score, p = m.learned_score(x, b)
    assert np.all((p >= 0.) & (p <= 1.) & np.isfinite(p))
    assert (score <= p[:, 0]).all()
