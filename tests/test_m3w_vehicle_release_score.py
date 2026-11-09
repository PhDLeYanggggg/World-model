import torch

from scripts.run_m3w_vehicle_fixed_release import selected
from src.world_model import m3w_vehicle_release_score as m


def test_nested_probability_cannot_exceed_the_risk_factor():
    x = torch.tensor([[-1., 3., 4.], [2., -3., 4.], [100., 100., 100.]])
    assert torch.all(m.log_score(x, True).exp() <= x[:, 0].sigmoid())
    torch.testing.assert_close(m.log_score(x, True).exp(), x.sigmoid().prod(dim=1))
    torch.testing.assert_close(m.log_score(x, False).exp(), x[:, 2].sigmoid())


def test_label_hierarchy_and_zero_support_minibatch_are_finite():
    y = torch.tensor([[0., 0., 3., 3., 4., 5.], [1., 0., 3., 3., 4., 5.],
                      [1., 1., 3., 3., -4., 5.], [1., 1., 3., 3., 4., 5.]])
    assert m.label_targets(y).int().tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    net = torch.nn.Linear(2, 6)
    for nested in (False, True):
        l = m.loss(net, (torch.zeros((1, 2)), y[:1]), nested)
        assert torch.isfinite(l)
        l.backward()


def test_frozen_release_keeps_missing_support_distinct_from_abstention():
    y = [(False, False, False, True, 0., 0.), (True, True, True, False, 2., 2.)]
    assert selected([.8, .9], y, None)['admitted'] == 0
    assert selected([.8, .9], y, .9)['all_full_win'] == 1
    assert selected([.8, .9], y, .8)['missing'] == 1
