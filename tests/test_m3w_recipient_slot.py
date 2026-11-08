import numpy as np
import pytest
import torch

from src.evaluation.m3w_recipient_slot_capacity import assess
from src.world_model import m3w_recipient_slot as slot


def test_radii_and_untouched_recipient():
    group = np.arange(3)
    edges, rows, amp = slot.feasible(group, [2., 1.], [2., 1.5, .5])
    np.testing.assert_array_equal(edges, [0, 1, 1])
    np.testing.assert_array_equal(rows, [0, 0, 1])
    np.testing.assert_allclose(amp, [1., .5, 2./3])
    with pytest.raises(ValueError, match="unedited"):
        slot.feasible(group[:2], [2., 1.], [2., 1.5, .5])


def test_joint_assignment_prevents_double_edit_and_retains_coverage_tradeoff():
    rows = np.tile(np.arange(2), 2)
    slots = np.repeat(np.arange(2), 2)
    value = np.array([10., 9., 10., 0.])
    prob = np.array([.05, .1, .1, .99])
    np.testing.assert_array_equal(slot.assign(rows, slots, value, prob, 2), [1, 2])
    np.testing.assert_array_equal(slot.assign(rows, slots, value, prob, 2, 20.), [0, 3])


def test_capacities_keep_undefined_and_do_not_drop_a_required_slot():
    cv = np.zeros((3, 12, 2))
    ols = cv.copy()
    y = np.full_like(cv, 1.)
    y[0] = .1
    valid = np.ones(cv.shape[:2], dtype=bool)
    cut = [.2, .9]
    assert assess(cv, ols, y, valid, [], [], [], cut) == "zero_edits"
    rows = np.array([1])
    slots = np.array([0])
    assert assess(cv, ols, y, valid, rows, slots, y[rows], cut) == "no_eligible_easy"
    rows = np.array([0, 1])
    assert assess(cv, ols, y, valid, rows, np.zeros(2, dtype=int), y[rows], cut) == "verified_positive_both"
    paths = y[rows].copy()
    paths[1] = -1.
    assert assess(cv, ols, y, valid, rows, np.arange(2), paths, cut) == "risk_infeasible"


def test_backpropagation_through_recipient_value_and_easy_event():
    g = np.random.default_rng(19)
    history = g.normal(size=(7, 8, 2)).cumsum(1)
    x, cv, ols, path, scale = slot.features(history, np.full(7, .5))
    y = g.normal(size=path.shape)
    y[-1] = np.nan
    valid = np.isfinite(y).all(2)
    target = slot.target(cv, ols, path, y, valid, 5., scale)
    assert target[-1, -1] == 0.
    assert np.isfinite(target).all()
    net = slot.Value(x.shape[1])
    loss = slot.objective(net, torch.from_numpy(x), torch.from_numpy(target))
    loss.backward()
    assert net.net[-1].bias.grad.abs().min() > 0.
