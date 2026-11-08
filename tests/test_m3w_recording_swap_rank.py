import numpy as np
import torch

from scripts.run_m3w_recording_swap_rank import batches
from src.world_model import m3w_recording_swap_rank as m


def test_cross_query_swaps_stay_in_recording_and_cannot_borrow_credit():
    legal = np.tile([True, True, False], (14, 1))
    rec = np.repeat(["a", "b"], 7)
    plan = np.zeros(14, dtype=int)
    plan[[4, 11]] = 1
    out = m.swaps([plan], legal, rec, np.random.default_rng(3), limit=20)
    assert out.tolist() == [[4, 5, 1], [4, 6, 1], [11, 12, 1], [11, 13, 1]]


def test_conservative_margin_is_zero_when_label_completion_can_flip_swap():
    parts = np.zeros((3, 3, 2))
    parts[0, 1] = [1., .2]
    parts[1, 1] = [2., .9]
    parts[2, 1] = [4., .1]
    edges = np.array([[0, 1, 1], [0, 2, 1], [2, 0, 1]])
    np.testing.assert_allclose(m.target(parts, edges, np.ones(3)), [0., 2.7, -2.7])


def test_ambiguous_swaps_do_not_change_the_point_loss():
    net = m.RankValue(4)
    x = torch.randn(2, 4)
    y = torch.zeros(2, 3, 2)
    legal = torch.tensor([[True, True, False]]*2)
    args = net, x, y, legal, x, x.flip(0), torch.ones(2, dtype=torch.long), torch.ones(2, 2), torch.zeros(2)
    torch.testing.assert_close(m.objective(*args, rank=False), m.objective(*args, rank=True))


def test_recording_batch_constructs_point_rows_and_cross_query_pairs():
    d = dict(x=np.ones((2, 4), np.float32), y=np.zeros((2, 3, 2), np.float32),
             legal=np.tile([True, True, False], (2, 1)), rows=np.arange(2),
             scale=np.ones(2), edges=np.array([[0, 1, 1]]), margin=np.array([.1], dtype=np.float32))
    args = batches([d])(np.random.default_rng(7))
    assert args[0].shape == (128, 4) and args[3].shape == (96, 4)
    loss = m.objective(m.RankValue(4), *args, rank=True)
    assert torch.isfinite(loss)
