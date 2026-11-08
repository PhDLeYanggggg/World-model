import numpy as np
import pytest
import torch

from src.world_model import m3w_recording_count as m
from scripts.audit_m3w_recording_count_matched import exact_score_plan


def test_causal_cap_is_prefix_invariant_and_resets_between_recordings():
    scores = np.tile([0., 1., 2.], (10, 1))
    legal = np.ones_like(scores, dtype=bool)
    rec = np.repeat(["a", "b"], 5)
    starts = np.tile(np.arange(5), 2)
    got = m.streaming_count(scores, legal, rec, starts, rate=.5)
    assert got.tolist() == [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    scores[3:] = [9., 0., 0.]
    np.testing.assert_array_equal(m.streaming_count(scores, legal, rec, starts, rate=.5)[:3], got[:3])
    with pytest.raises(ValueError, match="increasing"):
        m.streaming_count(scores[:2], legal[:2], rec[:2], [1, 1])


def test_teacher_chooses_funded_easy_support_over_larger_unsafe_gain():
    gains = np.array([[[0., 0.], [3., 3.], [0., 0.]], [[0., 0.], [1., 1.], [0., 0.]]])
    legal = np.array([[True, True, False], [True, True, False]])
    risk = np.zeros((2, 3, 3))
    risk[0, 1] = [1., 1., 1.]
    easy = np.array([[False, False, False], [False, True, False]])
    count, status = m.teacher(gains, risk, easy, legal, offset=0., easy_cap=.02, mass=20., budget=1)
    assert status == "verified_witness"
    assert count.tolist() == [0, 1]
    count, status = m.teacher(gains, risk, ~np.ones_like(easy), legal, offset=0., easy_cap=.02, mass=20., budget=1)
    assert status == "no_support"
    assert not count.any()


def test_count_head_never_decodes_unavailable_count():
    net = m.CountHead(4)
    score = net(torch.zeros(2, 4), torch.tensor([[True, False, False], [True, True, False]]))
    assert score.argmax(1)[0] == 0
    assert score.argmax(1)[1] <= 1


def test_retrospective_match_keeps_exact_total_but_cannot_borrow_future_tokens():
    score = np.zeros((10, 3))
    score[0, 2], score[9, 2] = 1000., 3.
    allowed = np.ones_like(score, dtype=bool)
    count = exact_score_plan(score, allowed, 2)
    assert count.tolist() == [0]*9+[2]
    assert not exact_score_plan(score, allowed, 0).any()
    with pytest.raises(ValueError, match="reachable"):
        exact_score_plan(score[:2], allowed[:2], 1)
