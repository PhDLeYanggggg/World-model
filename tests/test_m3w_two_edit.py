import numpy as np
import pytest
import torch

from scripts.replay_m3w_two_edit import normalize
from scripts.run_m3w_two_edit import finite_capacity
from src.world_model import m3w_two_edit as model


def test_candidate_pairs_respect_query_cap_and_disallow_duplicate_recipient():
    row = np.array([0, 0, 1, 2])
    q = np.array([0, 0, 0, 1])
    i, j = model.pairs(row, q, [1, 1])
    assert list(zip(i.tolist(), j.tolist())) == [(0, 3), (1, 3), (2, 3)]
    i, j = model.pairs(row, q, [2, 1])
    assert (0, 1) not in set(zip(i, j))
    assert (0, 2) in set(zip(i, j))


def test_joint_rank_can_use_complementary_predicted_gain_without_more_edits():
    v = np.zeros((3, 6))
    v[:, :2] = [[10., 0.], [0., 10.], [4., 4.]]
    i, j = model.pairs(np.arange(3), np.zeros(3, int), [2])
    got = model.rank(v, i, j, [100., 0., 0.])
    assert (i[got["joint"]], j[got["joint"]]) == (0, 1)
    assert j[got["independent"]] == j[got["joint_geometry"]] == 2


def test_easy_support_and_hard_gain_can_require_two_edits():
    y = np.zeros((3, 12, 2))
    cv = np.zeros_like(y)
    cv[:, :, 0] = np.array([1., 4., 1.])[:, None]
    ols = cv.copy()
    ols[1, :, 0] = 2.
    path = cv[:2].copy()
    path[:, :, 0] = np.array([.5, 1.])[:, None]
    v = np.ones((3, 12), bool)
    args = cv, ols, y, v, np.array([0, 1]), path
    summary = finite_capacity(*args, np.zeros(3, int), [2], [1.5, 3.])
    assert summary["pair"] and summary["same_query"] and summary["easy_and_hard"]
    assert summary["neither_single_wins"] and not summary["one"]
    assert not finite_capacity(*args, np.zeros(3, int), [1], [1.5, 3.])["pair"]


def test_shared_model_training_step_is_finite():
    net = model.network(5)
    y = torch.zeros(16, 6)
    loss = model.loss(net, torch.randn(16, 5), y)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_empty_short_view_is_null_for_all_arms_or_not_at_all():
    assert normalize({"short": {"joint": None, "independent": None}})["short"] is None
    v = {"short": {"joint": {"views": 1}, "independent": {"views": 1}}}
    assert normalize(v) == v
    with pytest.raises(ValueError):
        normalize({"short": {"joint": {"views": 1}, "independent": None}})
