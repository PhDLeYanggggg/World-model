import numpy as np
import pytest

from scripts.run_m3w_first_credit import one_edit
from src.world_model.m3w_first_credit import decode
from src.world_model.m3w_recording_count import streaming_count


def fixture(n=10):
    scores = np.tile([0., 1., 2.], (n, 1))
    return scores, np.ones_like(scores, dtype=bool), np.zeros(n), np.arange(n)


def test_advance_repays_before_next_edit_and_never_borrows_two():
    x = fixture()
    out = decode(*x, early_allowed=np.ones(10, bool))
    assert np.flatnonzero(out).tolist() == [0, 9]
    assert out.sum() == 2
    assert out[0] == 1


def test_no_gate_matches_existing_decoder_on_random_queries():
    rng = np.random.default_rng(817)
    n = 230
    score = rng.normal(size=(n, 3))
    legal = np.arange(3) <= rng.integers(0, 3, size=(n, 1))
    rec, start = np.arange(n)//23, np.arange(n) % 23
    np.testing.assert_array_equal(decode(score, legal, rec, start, early_allowed=np.zeros(n, bool)),
                                  streaming_count(score, legal, rec, start))


def test_causal_prefix_invariance_and_short_subsidy():
    args = fixture(10)
    full = decode(*args, early_allowed=np.ones(10, bool))
    prefix = decode(*(v[:3] for v in args), early_allowed=np.ones(3, bool))
    assert prefix.sum() == 1
    np.testing.assert_array_equal(prefix, full[:3])
    _, legal, *_ = args
    legal[:, 1:] = False
    assert not decode(*args, early_allowed=np.ones(10, bool)).any()


def test_recording_time_order_is_checked():
    scores, legal, rec, starts = fixture(3)
    with pytest.raises(ValueError):
        decode(scores, legal, rec, starts[::-1], early_allowed=[True]*3)


def test_one_edit_needs_easy_support_and_to_beat_both_references():
    y = np.zeros((2, 12, 2))
    cv = np.ones_like(y)
    valid = np.ones((2, 12), bool)
    path = cv[:1]*.5
    row = np.array([0])
    result = one_edit(cv, cv, y, valid, row, path, [2., 3.])
    assert result["both"] == 1
    assert one_edit(cv, y, y, valid, row, path, [2., 3.])["both"] == 0
    assert one_edit(cv, cv, y, valid, row, path, [1., 3.])["risk"] == 0
    valid[0] = False
    assert one_edit(cv, cv, y, valid, row, path, [2., 3.])["risk"] == 0
