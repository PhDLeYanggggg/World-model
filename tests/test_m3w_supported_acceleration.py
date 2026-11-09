import numpy as np
import torch

from scripts import run_m3w_opendd_supported_acceleration as run
from src.world_model import m3w_approach_candidate_paths as parent
from src.world_model import m3w_supported_acceleration_policy as model


def test_same_acceleration_as_av2():
    h = np.random.default_rng(7).normal(size=(4, 8, 2)).cumsum(1)
    cv, own, _, _ = parent.candidates(h, 12)
    bank = np.stack((cv, own[:, 1]), axis=1)
    p = run.with_acceleration(h, cv, bank)
    np.testing.assert_array_equal(p[:, :2], bank)
    np.testing.assert_array_equal(p[:, 2], own[:, 0])


def test_explicit_support_is_separate_from_predicted_positive_mass():
    g = np.array([[.01, .02], [.01, .02], [.01, .02], [.01, .02], [0., .02]])
    m = np.array([[0., 0., 0., 1., .4, 0.], [0., 0., 0., 1., .4, 0.],
                  [0., 0., 0., 1., 0., 0.], [0., 0., 0., 1., .4, .03], [0., 0., 0., 1., .4, 0.]])
    s = np.array([.01, .6, .99, .99, .99])
    np.testing.assert_array_equal(model.feasible(g, m, s, 1., 0., 0., .5), [False, True, False, False, False])
    np.testing.assert_array_equal(model.feasible(g, m, s, 1., 0., .6, .5), [True, True, False, False, False])
    np.testing.assert_array_equal(model.feasible(g, m, s, 1., 0., 0., 0.), [True, True, False, False, False])


def test_support_head_cannot_depend_on_candidate_shape_or_radius():
    torch.manual_seed(11)
    net = model.SupportedCost().eval()
    h = torch.zeros((2, 45))
    h[0, 20:] = torch.arange(25)
    with torch.inference_mode():
        gain, moments, support = net(h, torch.tensor([2., 0.]))
    assert gain.shape == (2, 2)
    assert moments.shape == (2, 6)
    assert support[0] == support[1]
    assert moments[0, 4] == moments[1, 4]


def test_singleton_does_not_add_a_dummy_actor_cost():
    x = np.array([[.9, .7], [.1, .2]])
    opts = np.array([[0, -1], [1, 0]])
    np.testing.assert_allclose(model.packed(x, opts).sum(1), [[.9, .7], [1., .9]])
