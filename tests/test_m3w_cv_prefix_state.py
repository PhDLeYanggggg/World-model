import numpy as np
import torch

from scripts import train_m3w_cv_prefix_state as study
from src.world_model import m3w_cv_prefix_state as model


def test_the_same_edit_changes_risk_differently_with_prefix_balance():
    c = np.zeros((2, 8))
    c[0, 4] = -.3
    c[1, 2:5] = [.4, .5, .4]
    np.testing.assert_allclose(model.marginal(np.zeros(8), c)[1], [.4, .5, .4])
    prefix = np.zeros(8)
    prefix[2:5] = [-.3, -.4, -.5]
    np.testing.assert_allclose(model.marginal(prefix, c)[1], [.1, .1, 0.])
    y = model.target(np.zeros((2, 6)), prefix, c, 1., 1.)
    np.testing.assert_allclose(y[1, 2:5], [.1, .1, 0.])


def test_shadow_state_never_reads_later_queries_or_outcomes_and_resets():
    def query(rec, shift):
        d = np.zeros((2, 8))
        d[1, :2] = shift
        return dict(rec=rec, x=np.zeros((2, 2), np.float32), at=[0],
            anchor=1, scale=1., predicted=d)
    a = [query(0, 2.), query(0, 3.), query(1, 4.)]
    b = [query(0, 2.), query(0, 100.), query(1, 4.)]
    study.shadow_packet(a)
    study.shadow_packet(b)
    np.testing.assert_array_equal(a[0]['state_x'], b[0]['state_x'])
    np.testing.assert_array_equal(a[1]['state_x'], b[1]['state_x'])
    np.testing.assert_array_equal(a[0]['state_x'], a[2]['state_x'])
    assert (a[1]['state_x'] != a[0]['state_x']).any()


def test_masked_input_is_immune_to_prefix_values():
    torch.manual_seed(1)
    x = torch.randn(3, 2, 2 + model.WIDTH)
    y = torch.randn(3, 2, 6)
    net = torch.nn.Linear(x.shape[-1], 6)
    z = x.clone()
    z[..., -model.WIDTH:] += 4.
    assert model.objective(net, x, y, False) == model.objective(net, z, y, False)
    assert model.objective(net, x, y, True) != model.objective(net, z, y, True)


def test_reference_mass_and_count_support_are_separate():
    g = np.ones((2, 2, 2))
    r = np.zeros((2, 2, 3))
    ref = np.array([[1., 1.], [2., 0.]])
    a = np.array([[False, False], [False, True]])
    z = study.summaries(g, r, ref, a)
    assert z[1, 5] == 0.
    np.testing.assert_allclose(z[:, 6:], [[3., 1.], [3., 1.]])
    np.testing.assert_allclose(z[:, 4], [-.02, -.02])
