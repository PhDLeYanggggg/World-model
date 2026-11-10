import numpy as np
import pytest

from src.evaluation.m3w_seed_shared_geometry import shared_geometry


def test_reversed_seed_order_has_no_effect_for_one_shared_target():
    a = np.zeros((3, 1, 2, 2))
    a[:, 0, 0, 0] = [0, 1, 2]
    d = shared_geometry(a, a[::-1], np.array([[True, False]]), np.array([.5]))
    assert d['per_seed_diagonal'] == [1., 0., 1.]
    assert d['shared'] == d['witness_low'] == d['witness_high'] == 0.
    assert d['affected_steps'] == d['tighter_steps'] == 1


def test_identical_seed_directions_retain_the_diagonal_and_attain_both_signs():
    a = np.zeros((3, 1, 1, 2))
    b = np.ones_like(a)
    d = shared_geometry(a, b, np.ones((1, 1), bool), np.array([.25]))
    assert d['shared'] == pytest.approx(np.sqrt(2.)*.25)
    assert d['witness_low'] == -d['witness_high'] == -d['shared']
    assert d['tighter_steps'] == 0


def test_unaffected_mask_and_invalid_weight():
    a = np.zeros((3, 1, 1, 2))
    d = shared_geometry(a, a, np.zeros((1, 1), bool), np.array([1.]))
    assert d['shared'] == 0. and d['affected_steps'] == 0
    with pytest.raises(ValueError, match='original query weights'):
        shared_geometry(a, a, np.zeros((1, 1), bool), np.array([-1.]))
