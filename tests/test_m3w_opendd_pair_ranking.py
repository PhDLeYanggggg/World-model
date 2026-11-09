import numpy as np
import torch

from scripts import run_m3w_opendd_pair_ranking as run
from scripts import audit_m3w_pair_ranking_targets as audit


def test_mining_prefers_supported_negatives_within_the_query():
    x = np.zeros((4, 2, 45))
    g = np.repeat(np.array([[0.], [.001], [.1], [1.]]), 8, axis=1)
    y = np.array([[1., 1.], [0., 0.], [0., 1.], [0., 1.]])
    rng = np.random.default_rng(4)
    np.testing.assert_array_equal(run.mine(x, g, y, rng), [[0, 2], [0, 3]])
    assert run.mine(x, g, np.zeros((4, 2)), rng).shape == (0, 2)


def test_pairwise_loss_rewards_the_order_not_the_common_offset():
    x = torch.zeros(2, 2, 2, 2)
    x[:, 0, :, 0] = 3.
    torch.testing.assert_close(run.ranking_loss(x), run.ranking_loss(x+2.))
    assert run.ranking_loss(x) < run.ranking_loss(x.flip(1))


def test_same_label_tensor_broadcasts_to_both_heads_and_mined_options():
    a = torch.zeros(4, 2, 2)
    b = torch.zeros(4, 2)
    torch.testing.assert_close(run.bce(a, b), run.bce(a[:, None].expand(-1, 2, -1, -1),
                                                      b[:, None].expand(-1, 2, -1)))


def test_pair_target_requires_a_strict_gain_over_the_additive_path():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[:2, :, 0] = np.array([1., .5])[:, None]
    ref = .3*y
    v = np.ones(y.shape[:2], bool)
    feasible, joint = audit.targets(dict(c=cv, ols=cv), np.stack((y, ref, cv)), ref, y, v, 2.)
    np.testing.assert_array_equal(feasible, [True, True, False])
    np.testing.assert_array_equal(joint, [True, False, False])
