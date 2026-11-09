import numpy as np
import torch

from src.world_model import m3w_av2_projection_risk as model
from src.world_model import m3w_av2_recipient_reassignment as bank
from scripts.audit_m3w_av2_projection_risk_matched import weights


def test_projection_targets_replay_exact_partial_label_eligibility():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 1., 3., .8])[:, None]
    valid = np.ones(cv.shape[:-1], dtype=bool)
    valid[0] = False
    valid[1, -2:] = False
    p = np.stack((y*.7, y*.85), axis=1)
    opts = bank.assignments(len(cv))
    gain, labels = model.targets(dict(cv=cv, ols4=cv), p, opts, y, valid, 2., 1.)
    expected, _ = bank.eligibility(cv, cv, p, y, valid, 2., opts)
    np.testing.assert_array_equal(labels.all(1), expected)
    np.testing.assert_array_equal((gain > 0.).all(1), labels[:, 2])


def test_risk_probability_is_only_supervised_on_supported_rows():
    x = torch.zeros(2, 4, requires_grad=True)
    status = torch.tensor([[False, False, False], [True, True, True]])
    model.loss(x, torch.ones(2, 2), status)[0].backward()
    assert x.grad[0, 3] == 0. and x.grad[1, 3] != 0.
    a, b = model.scores(np.array([[.1, .2, 8., -8.], [.1, .2, 8., 8.]]))
    assert a[0] == a[1] and b[0] < b[1]


def test_same_count_diagnostic_shares_tied_slots_without_labels():
    w = weights(np.array([.9, .5, .5, .1]), 2)
    np.testing.assert_array_equal(w, [1., .5, .5, 0.])
    assert w.sum() == 2
