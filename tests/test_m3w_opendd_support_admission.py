import numpy as np
import pytest
import torch

from src.world_model import m3w_opendd_support_admission as m


def test_menu_requires_old_trigger_but_can_consider_outside_easy():
    easy = np.array([.95, .4, .6])
    rows, anchor = m.menu(np.arange(3), easy, first=0)
    np.testing.assert_array_equal(rows, [1, 2])
    assert anchor == 2
    assert m.menu(np.array([0, 1]), easy, first=0)[1] is None
    assert m.menu(np.array([], dtype=int), easy)[1] is None


def test_guard_falls_back_on_predicted_gain_loss_or_harm():
    easy = np.array([.6, .4, .3])
    rows = np.arange(3)
    z = np.array([[2., .2, .5], [1., .9, .4], [3., .8, .6]])
    pick = m.choose(rows, 0, easy, z)
    assert pick == dict(local_gain=2, local_support=1, restricted_support=0, guarded_support=0)
    z[1, 0] = 2.1
    assert m.choose(rows, 0, easy, z)["guarded_support"] == 1
    z[1, 0] = np.nan
    with pytest.raises(ValueError):
        m.choose(rows, 0, easy, z)


def test_anchor_comparison_uses_only_current_menu():
    x = np.array([[2., 3.], [3., 5.], [4., 6.]])
    easy = np.array([.6, .2, .3])
    rows = np.array([0, 2])
    f = m.features(x, easy, rows, 0)
    x[1] = 1e9
    np.testing.assert_array_equal(m.features(x, easy, rows, 0), f)
    np.testing.assert_array_equal(f[:, 2:4], [[0., 0.], [2., 3.]])
    net = m.network(f.shape[1])
    value = m.loss(net, torch.from_numpy(f), torch.tensor([[0., 0., 0.], [1., 1., .2]]))
    value.backward()
    assert torch.isfinite(value)
