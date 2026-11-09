import numpy as np
import pytest

from scripts import run_m3w_opendd_joint_retrieval as run
from scripts import audit_m3w_joint_retrieval_debit as audit


def test_joint_retrieval_reserves_feasible_choices_and_uses_both_scores():
    feasible = np.arange(12., 0., -1.)
    pair = np.r_[np.zeros(8), [1., 2., 3., 4.]]
    old, new = run.retrieve(feasible, pair)
    np.testing.assert_array_equal(old, np.arange(8))
    np.testing.assert_array_equal(new[:4], old[:4])
    assert set(new[4:]) == set(range(8, 12))
    assert len(set(new)) == len(new) == run.WIDTH


def test_retrieval_is_deterministic_with_ties_and_short_menus():
    both = np.zeros(3)
    old, new = run.retrieve(both, both)
    np.testing.assert_array_equal(old, [0, 1, 2])
    np.testing.assert_array_equal(old, new)


def test_retrieval_rejects_missing_scores():
    with pytest.raises(ValueError, match="finite"):
        run.retrieve(np.ones(2), np.array([0., np.nan]))


def test_missing_debit_and_observed_gain_share_the_full_grid_denominator():
    before = np.zeros((2, 2, 2))
    after = before.copy()
    after[0, :, 0] = 1.
    valid = np.array([[True, False], [True, True]])
    terms = audit.components(after, before, after, valid, np.zeros(2, int))
    np.testing.assert_allclose(terms, [.5, .5, 0.])
