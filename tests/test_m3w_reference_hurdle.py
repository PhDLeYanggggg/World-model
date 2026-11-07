import numpy as np
import torch

from scripts import run_m3w_reference_hurdle as runner
from src.world_model import m3w_reference_hurdle as model


def test_targets_charge_missing_label_disagreement_without_imputing_support():
    atoms = np.zeros((2, 1, 6))
    atoms[:, 0, 1] = [2., 0.]
    atoms[:, 0, -1] = [.5, -5.]
    t = model.targets(atoms, np.array([.5, 2.]))
    np.testing.assert_allclose(t[:, 0, 0], [1., 0.])
    np.testing.assert_allclose(t[:, 0, 1], [1., -2.5])
    np.testing.assert_allclose(t[:, 0, 2], [0., 2.5])
    out = torch.zeros((2, 4), requires_grad=True)
    model.objective(out, torch.from_numpy(t[:, 0]).float()).backward()
    assert torch.isfinite(out.grad).all()
    assert out.grad[1, 1] == out.grad[0, 2] == 0.


def test_hurdle_and_direct_order_identical_candidate_set_differently():
    direct = np.array([-1, -1, 0, -1, -1, -1])
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    legal = np.ones((6, 1), bool)
    scores = np.array([[[.9, .1, .1]], [[.8, .3, .2]], [[.9, 1., 1.]],
                       [[.9, 1., 1.]], [[.7, .4, .3]], [[.6, .5, .4]]])
    assert model.first(list(model.options(direct, groups, legal, scores, 'support')), (.5, 0.)) == (0, 0)
    opts = list(model.options(direct, groups, legal, scores, 'hurdle'))
    assert model.first(opts, (.5, 0.)) == (1, 0)
    assert model.first(opts, (.5, .25)) == (5, 0)
    assert model.first(opts, None) is None


def test_calibration_requires_actual_support_and_preserves_other_recordings():
    unsupported = dict(supported=False, easy_net_over=False)
    atom = np.array([[[1., 1., 0., 0., -1., 1.]]])
    donors = [([(0, 0, .6, .3)], atom, np.zeros(6), 1., np.ones(1), unsupported)]
    assert runner.calibrate(donors) is not None
    bad = atom.copy()
    bad[0, 0, 1] = 0.
    assert runner.calibrate([(donors[0][0], bad, *donors[0][2:])]) is None
    harm = np.array([[[1., 1., 1., 1., 1., -1.]]])
    assert runner.calibrate(donors + [(donors[0][0], harm, *donors[0][2:])]) is None
