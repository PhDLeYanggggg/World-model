import numpy as np
import torch

from scripts import train_m3w_cv_risk_crossing as study
from src.world_model import m3w_cv_risk_crossing as model


def test_old_witness_paths_map_by_fraction_and_ray():
    old = np.array([-1, 0, 1, 2, 6, 7])
    a = model.to_current(old, (.05, .15, .35, 1.), (.05, .15, .35, .65, 1.), 2)
    np.testing.assert_array_equal(a, [-1, 0, 5, 1, 4, 9])


def test_boundary_pairs_reproduce_original_recording_check():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[..., 0] = np.array([1., 10., 10.])[:, None]
    valid = np.ones((3, 12), bool)
    p = np.zeros((3, 2, 12, 2))
    p[0, :, :, 0] = np.array([.1, 2.1])[:, None]
    p[1, :, :, 0] = 5.
    p[2, :, :, 0] = -3.
    q = np.array([0, 0, 0])
    a = np.array([0, 0, -1])
    terms, required = study.quantities(cv, cv, p, y, valid, q, 2.)
    keep, label, harm = model.crossings(terms, required, a,
        np.array([0, 0]), np.array([0, 0]), np.array([0, 2]), np.array([1, 0]))
    np.testing.assert_array_equal(keep, [True, False])
    assert label[0] == 1
    assert harm[0, 1]
    scorer = study.prior.prior.parent.readout
    for action, expected in ((a, True), (np.array([1, 0, -1]), False)):
        forecast = cv.copy()
        at = np.flatnonzero(action >= 0)
        forecast[at] = p[at, action[at]]
        z = scorer.score_recording({study.base.CV: cv, 'p': forecast}, {},
            y, valid, q, 'toy', easy_cut=2., hard_cut=7.)
        assert z['metrics']['p']['risk']['supported'] == expected


def test_recipient_exchange_preserves_query_and_excludes_existing_recipients():
    a = np.array([0, -1, -1, 0, -1])
    l = np.ones((5, 2), bool)
    l[2] = False
    h = np.zeros((5, 8, 2))
    q = [np.array([0, 1, 2]), np.array([3, 4])]
    i, ai, j, aj = study.candidates(a, l, h, q)
    assert (ai == 0).all()
    assert set(zip(i[i != j], j[i != j])) == {(0, 1), (3, 4)}
    assert (l[j, aj]).all()


def test_risk_and_utility_objectives_disagree_when_larger_gain_crosses_limit():
    x = torch.tensor([[[2.], [0.]]])
    y = torch.tensor([-1.])
    assert model.objective(torch.nn.Identity(), x, y, True) < model.objective(torch.nn.Identity(), x, y, False)
