import numpy as np
import torch

from src.world_model.m3w_conditional_gain import combined, common_gaps, matched_choices, support_layout


def test_common_pair_ignores_nonoverlap_and_wholly_missing_pair():
    labels = torch.tensor([[[.8, -.9, .2], [.4, .8, 1.]], [[1., 1., 1.], [1., 1., 1.]]])
    visible = torch.tensor([[[1, 1, 0], [1, 0, 1]], [[1, 0, 0], [0, 1, 0]]], dtype=torch.bool)
    pred, real = common_gaps(labels/2, labels, torch.ones_like(labels), visible)
    torch.testing.assert_close(real, torch.tensor([.4, 0.]))
    torch.testing.assert_close(pred, torch.tensor([.2, 0.]))


def test_fixed_visibility_bins_preserve_reference_counts():
    p = np.linspace(1., 0., 9)
    eligible = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], bool)
    ref = np.array([0, 0, 0, 2, 3, 0, 0, 0, 10])
    layout = support_layout(p, ref, eligible, [np.arange(len(p))])
    a = matched_choices(ref, layout)
    b = matched_choices(-ref, layout)
    assert [a[q].sum() for q, _ in layout] == [0, 1, 1, 0]
    assert [b[q].sum() for q, _ in layout] == [0, 1, 1, 0]
    assert not np.array_equal(a, b)
    assert not b[8]


def test_support_debit_is_shared_by_all_gain_heads():
    mu = np.array([[.3, .8]])
    p = np.array([[1., 0.]])
    d = np.array([[2., 4.]])
    np.testing.assert_allclose(combined(mu, p, d), [-1.7])
