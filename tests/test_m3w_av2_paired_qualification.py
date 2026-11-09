import numpy as np
import torch

from scripts import run_m3w_av2_paired_qualification as run
from src.world_model import m3w_av2_paired_qualification as model


def test_query_weight_changes_release_confidence_not_candidate_order():
    z = torch.tensor([[3., -4.], [1., 2.], [-5., 0.]])
    high, qh = model.probabilities(z, torch.tensor(3.))
    low, ql = model.probabilities(z, torch.tensor(-3.))
    assert int(high["direct"].argmax()) == int(low["query_weighted"].argmax()) == 0
    assert int(low["conditional"].argmax()) == 1
    assert qh > ql and high["conditional"].max() > low["conditional"].max()
    torch.testing.assert_close(low["query_weighted"], ql*low["direct"])


def test_same_query_loss_prefers_higher_positive_logit():
    good = torch.tensor([[2., 3.], [2., 1.], [-1., -3.], [-1., -2.]])
    bad = good.roll(2, dims=0)
    for h in (0, 1):
        assert model.ranking(good, h) < model.ranking(bad, h)


def test_exclude_all_reference_copies_but_keep_anchor_for_query_features():
    ref = np.zeros((2, 12, 2))
    p = np.stack((ref, ref+1., ref.copy(), ref+3.))
    x = np.arange(8).reshape(4, 2)
    s = (p, ref, x, np.ones(4), np.zeros((4, 3)), np.zeros((4, 4)), None, 1.)
    features, a, keep = run.features(s)
    np.testing.assert_array_equal(keep, [1, 3])
    torch.testing.assert_close(a, features[0])
