import numpy as np
import torch

from scripts import run_m3w_av2_relative_margins as run
from src.world_model import m3w_av2_relative_margins as model


def test_missing_easy_label_does_not_train_a_margin():
    z = torch.zeros(2, 7, requires_grad=True)
    y = torch.tensor([[0., 1., 0., 1., 1., 1., -1.], [1., 1., 1., 1., 1., 1., 1.]])
    model.loss(z, y, dense=True).backward()
    assert z.grad[0, 6] == 0.
    assert z.grad[1, 6] < 0.
    assert z.grad[0, 3] < 0.


def test_relative_trajectory_and_risk_can_change_a_binary_rank():
    z = torch.tensor([[.2, 1., 1., .2, -.2, .2, .2],
                      [.1, 1., 1., .2, .2, .2, .2]])
    assert model.scores(z, dense=False).argmax() == 0
    assert model.scores(z, dense=True).argmax() == 1


def test_margins_reproduce_the_exact_matched_labels_with_missing_steps():
    cv = np.zeros((3, 12, 2))
    a = cv.copy()
    a[0, :, 0], a[1, :, 0] = 1., 2.
    p = np.stack((a, a*.5, a*2., cv))
    d = dict(cv=cv, ols4=cv.copy())
    y = a.copy()
    y[2] += .02
    v = np.ones((3, 12), dtype=bool)
    v[1, -1] = False
    cut = [1.1, 2.5]
    s = (p, a, None, 2.)
    label, eligible = run.pair.truth_terms(d, p, a, y, v, cut)
    out = run.targets(d, s, y, v, cut)
    np.testing.assert_array_equal(out[:, 0], label & eligible)
    assert eligible
    assert out[0, 3] == out[0, 4] == 0.
    assert out[1, 4] < 0.
    assert out[-1, 2] == 0.


def test_wholly_missing_unedited_actor_does_not_poison_a_valid_candidate():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[0, :, 0], y[1, :, 0] = .8, 2.
    a = cv.copy()
    a[0, :, 0], a[1, :, 0] = .6, 1.6
    bad = y.copy()
    bad[3, :, 0] = .5
    v = np.ones((4, 12), dtype=bool)
    v[3] = False
    cut = [1.1, 2.5]
    d = dict(cv=cv, ols4=cv.copy())
    p = np.stack((y, bad))
    s = (p, a, None, 1.)
    t = run.targets(d, s, y, v, cut)
    corrected, legacy, eligible, diff = run.audit_labels(d, s, y, v, cut, t, [0, 1])
    assert eligible
    np.testing.assert_array_equal(legacy, [False, False])
    np.testing.assert_array_equal(corrected, [True, False])
    np.testing.assert_array_equal(diff, [0])
    assert np.isfinite(t).all()
