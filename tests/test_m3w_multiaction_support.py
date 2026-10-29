import itertools
import numpy as np
import pytest

from src.evaluation.m3w_multiaction_support import solve_action_bank, score_choices
from src.evaluation.m3w_easy_constrained_policy import budget_status
from src.world_model.m3w_fixed_floor_probe import targets


def test_mutually_exclusive_oracle_matches_small_exhaustive_search():
    rng = np.random.default_rng(271828)
    for _ in range(80):
        n, k = 3, 2
        reference = rng.uniform(.2, 5., n)
        cv = rng.choice([.5, 2.], n)
        candidate = rng.uniform(0., 7., (n, k))
        tied = rng.random((n, k)) < .25
        candidate[tied] = np.broadcast_to(reference[:, None], (n, k))[tied]
        y = np.stack([targets(cv, reference, candidate[:, a], 1.) for a in range(k)], axis=1)
        envelope = reference[:, None]+candidate
        allowed = rng.random((n, k)) > .2
        result, choices = solve_action_bank(y, envelope, allowed)
        brute = []
        for choice in itertools.product(range(-1, k), repeat=n):
            choice = np.array(choice)
            if any(a >= 0 and not allowed[i, a] for i, a in enumerate(choice)):
                continue
            score = score_choices(y, envelope, choice)
            if budget_status(score)['supported']:
                brute.append(score['selected_net_gain_lower_mass'])
        assert result['budget_feasible'] == bool(brute)
        if brute:
            assert result['optimal_gain_lower_mass'] == pytest.approx(max(brute), abs=1e-10)
            assert (choices >= -1).all() and (choices < k).all()


def test_noop_exclusion_and_shared_reference_required():
    y = np.array([[[0., 0., 1., 1., 0.], [0., 2., 1., 1., 2.]]])
    env = np.array([[0., 2.]])
    result, choices = solve_action_bank(y, env, np.array([[False, True]]))
    assert not result['budget_feasible'] and choices[0] == -1
    assert result['policy']['easy_selected_risk_upper'] is None
    y[0, 1, 2] = 2.
    with pytest.raises((ValueError, AssertionError)):
        solve_action_bank(y, env, np.array([[False, True]]))


def test_unknown_outcomes_cannot_be_oracle_gain_or_zero_risk():
    y = np.array([[[1., 0., 2., 2., 0.], [0., 1., 2., 2., 1.]],
                  [[np.nan]*5, [np.nan]*5]])
    result, choices = solve_action_bank(y, np.ones((2, 2)), np.ones((2, 2), bool))
    np.testing.assert_array_equal(choices, [0, -1])
    assert result['policy']['unknown_rows'] == 1
    assert result['policy']['selected_unknown'] == 0
    assert result['optimal_gain_lower_mass'] == 1.
