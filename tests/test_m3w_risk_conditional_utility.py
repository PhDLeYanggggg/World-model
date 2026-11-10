import numpy as np
import torch

from scripts.run_m3w_opendd_risk_conditional_utility import batcher
from src.evaluation.m3w_risk_conditional_utility import neighbors


def test_neighbors_keep_count_support_and_movement_and_require_lower_utility():
    n = 10
    legal = np.tile([True, True, False], (n, 1))
    movement = np.tile([0., 1., 1.], (n, 1))
    movement[5, 1] = 1.5
    support = legal.copy()
    support[:, 0] = False
    support[9, 1] = False
    risk = np.zeros((n, 3, 3))
    risk[4, 1, 0] = 1.
    gain = np.zeros((n, 3, 2))
    gain[:, 1] = np.arange(n)[:, None]
    gain[7, 1] = 10.
    good = np.zeros(n, dtype=int)
    good[7] = 1
    options = neighbors(good, movement, 1., (gain, risk, support, legal, np.zeros(2), .02, 100.))
    assert [p.nonzero()[0].tolist() for p in options] == [[6], [8]]
    assert all(p.sum() == 1 for p in options)


def test_first_batch_draws_are_identical_when_the_auxiliary_pairs_change():
    counts = np.array([[1, 0], [0, 1]])
    mask = np.ones((2, 3), dtype=bool)
    item = lambda x: (np.full((2, 1), x, dtype=np.float32), mask, counts, 1.)
    ordinary = [[item(1), item(2)], [item(3)]]
    conditional = [[item(4), item(5), item(6)]]
    control = batcher(ordinary, ordinary)
    treatment = batcher(ordinary, conditional)
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(4):
        x, y = control(a), treatment(b)
        for first, second in zip(x[:5], y[:5], strict=True):
            torch.testing.assert_close(first, second)
        assert not torch.equal(x[5], y[5])
