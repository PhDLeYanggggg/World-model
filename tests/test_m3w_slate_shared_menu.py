import numpy as np
import pytest
import torch

from src.world_model.m3w_slate_shared_menu import choose, legal_assignments, slot_menu


class DistanceScore(torch.nn.Module):
    def forward(self, x, pair, on):
        unary = torch.zeros(len(x))
        return unary, unary + pair[:, 0]*on.all(-1)


def paths():
    h = np.zeros((3, 8, 2))
    h[:, :, 0] = np.arange(8)[None] + np.array([0., 2., 50.])[:, None]
    cv = np.zeros((3, 12, 2))
    cv[:, :, 0] = np.arange(1, 13)
    bank = cv[:, None].copy()
    bank[:, :, :, 1] = np.arange(1, 13)*.25
    return h, cv, bank


def test_two_slots_use_distinct_recipients():
    options = legal_assignments(np.array([0, 1]), np.array([1, 2]))
    assert options.tolist() == [[0, 2], [0, 3], [1, 3]]


def test_pair_term_can_change_choice_but_not_budget():
    h, cv, bank = paths()
    a, b, n = choose(DistanceScore(), h, cv, cv, bank, [1., 2.])
    assert n == 6
    assert not np.array_equal(a, b)
    for output in (a, b):
        peak = np.linalg.norm(output-cv, axis=-1).max(axis=-1)
        assert sorted(peak) == [0., 1., 2.]


def test_one_slot_exact_null_control():
    h, cv, bank = paths()
    a, b, n = choose(DistanceScore(), h, cv, cv, bank, [1.])
    assert n == 3
    np.testing.assert_array_equal(a, b)


def test_menu_shortens_legal_candidates_never_enlarges_them():
    _, cv, bank = paths()
    bank[0, :, :, 1] *= .1
    p, ((rows, arms),) = slot_menu(cv, bank, [1.])
    assert rows.tolist() == [1, 2]
    np.testing.assert_array_equal(p[0, 0], cv[0])
    np.testing.assert_allclose(np.linalg.norm(p[rows, arms]-cv[rows], axis=-1).max(axis=1), 1.)


def test_fixed_budget_cannot_edit_whole_roster():
    _, cv, bank = paths()
    with pytest.raises(ValueError, match='one CV recipient'):
        slot_menu(cv[:2], bank[:2], [1., 1.])
