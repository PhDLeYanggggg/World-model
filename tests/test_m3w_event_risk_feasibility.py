import itertools

import numpy as np
import pytest

from src.world_model.m3w_event_risk_feasibility import bounded_risk_coefficients
from src.world_model.m3w_joint_intervention import InterventionProblem
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control


def test_extreme_impossible_ratio_is_pruned_without_epsilon():
    out = bounded_risk_coefficients(np.array([1.]), np.array([[1e-21, 1.]]),
        np.ones(1, bool), budget=.02, support_available=True)
    assert not out['supported'].any() and not out['expected_harm'].any()
    p = InterventionProblem(out['expected_gain'], out['expected_harm'], out['supported'],
        np.empty((0, 2), int), np.empty((0, 2, 2)), .1, .02, 1)
    solved = solve_scaled_risk_control(p, objective_kind='joint', time_limit_seconds=2.)
    assert solved['solver_optimal'] and not solved['switch'].any()


def test_pruning_preserves_every_feasible_subset():
    rng = np.random.default_rng(39271)
    for _ in range(80):
        moments = 10.**rng.uniform(-20, 6, size=(5, 2))
        utility = rng.normal(size=5)
        moving = rng.random(5) > .2
        out = bounded_risk_coefficients(utility, moments, moving, budget=.02, support_available=True)
        original_supported = moving & (utility > 0)
        for subset in itertools.product((False, True), repeat=5):
            b = np.asarray(subset)
            original = bool(not np.any(b & ~original_supported) and np.dot(b, moments[:, 1]) <= .02*moments[:, 0].sum())
            repaired = bool(not np.any(b & ~out['supported']) and np.mean(b*out['expected_harm']) <= .02)
            assert repaired == original
        assert (out['expected_harm'] <= .02*len(utility)*(1+1e-12)).all()


def test_missing_support_and_denominator_remain_fallback():
    for support in (True, False):
        out = bounded_risk_coefficients(np.ones(2), np.zeros((2, 2)), np.ones(2, bool),
            budget=.02, support_available=support)
        assert not out['supported'].any()
    with pytest.raises(ValueError):
        bounded_risk_coefficients(np.ones(2), np.ones((2, 2)), np.ones(2, bool),
                                 budget=-1, support_available=True)
