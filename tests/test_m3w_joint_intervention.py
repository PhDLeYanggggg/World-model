from itertools import product

import numpy as np
import pytest

from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table, select_interventions,
    realized_relative_costs, screen_cluster_risks,
)


def problem(gain=(0.8, -0.1), harm=(0.0, 0.0), penalty=2.0):
    # Each whole forecast is separated, but selecting only agent 0 co-locates both.
    baseline = np.array([[[0., 0.]], [[2., 0.]]])
    candidate = np.array([[[2., 0.]], [[4., 0.]]])
    edges = np.array([[0, 1]], dtype=int)
    costs = proximity_cost_table(baseline, candidate, edges, distance_threshold=1.0)
    return InterventionProblem(np.array(gain), np.array(harm), np.ones(2, dtype=bool),
                               edges, costs, pair_weight=penalty,
                               max_mean_predicted_harm=0.1, max_interventions=2)


def test_joint_selection_rejects_incompatible_independent_mix():
    p = problem()
    independent = select_interventions(p, mode="independent")
    joint = select_interventions(p, mode="joint")
    np.testing.assert_array_equal(independent["switch"], [True, False])
    np.testing.assert_array_equal(joint["switch"], [True, True])
    assert joint["mean_pair_proxy"] < independent["mean_pair_proxy"]
    assert joint["objective"] < independent["objective"]
    assert joint["solver_optimal"] and not joint["physical_safety_certified"]


def test_all_controls_and_support_gate_are_explicit():
    p = problem()
    np.testing.assert_array_equal(select_interventions(p, mode="floor")["switch"], [False, False])
    np.testing.assert_array_equal(select_interventions(p, mode="uncontrolled")["switch"], [True, True])
    np.testing.assert_array_equal(select_interventions(p, mode="scene_uniform")["switch"], [True, True])
    p.supported[1] = False
    assert not select_interventions(p, mode="joint")["switch"].any()
    assert not select_interventions(p, mode="scene_uniform")["switch"].any()


def test_predicted_harm_budget_and_cardinality_are_enforced():
    p = problem(gain=(1., 1.), harm=(0.4, 0.0), penalty=0.)
    p.max_interventions = 1
    np.testing.assert_array_equal(select_interventions(p, mode="joint")["switch"], [False, True])
    p.expected_gain[:] = -1
    assert not select_interventions(p, mode="joint")["switch"].any()


def test_milp_matches_exhaustive_search_on_random_small_scenes():
    rng = np.random.default_rng(42)
    for _ in range(30):
        n = 5
        edges = np.array(list((i, j) for i in range(n) for j in range(i+1, n)))
        cost = rng.uniform(0, 1, (len(edges), 2, 2))
        cost[:, 0, 0] = 0
        p = InterventionProblem(rng.normal(size=n), rng.uniform(0, .3, n),
                                rng.random(n) > .2, edges, cost, pair_weight=.7,
                                max_mean_predicted_harm=.1, max_interventions=3)
        feasible = []
        for bits in product((0, 1), repeat=n):
            b = np.array(bits)
            if (b <= p.supported).all() and b.sum() <= 3 and (b*np.maximum(p.expected_harm, -p.expected_gain)).mean() <= .1:
                pair = np.mean([cost[e, b[i], b[j]] for e, (i, j) in enumerate(edges)])
                feasible.append(-(b*p.expected_gain).mean() + .7*pair)
        result = select_interventions(p, mode="joint")
        assert result["objective"] == pytest.approx(min(feasible), abs=1e-8)


def test_solver_failure_returns_floor_not_unverified_incumbent(monkeypatch):
    from types import SimpleNamespace
    import src.world_model.m3w_joint_intervention as mod
    monkeypatch.setattr(mod, "milp", lambda **kwargs: SimpleNamespace(success=False, status=1))
    result = select_interventions(problem(), mode="joint")
    assert not result["switch"].any()
    assert result["reason"] == "solver_not_optimal_floor"
    assert not result["solver_optimal"]


def test_negative_expected_gain_implies_a_positive_harm_floor():
    p = problem()
    p.max_mean_predicted_harm = .01
    # The only geometrically coherent switch includes negative expected gain.
    assert not select_interventions(p, mode="joint")["switch"].any()
    uncontrolled = select_interventions(p, mode="uncontrolled")
    assert uncontrolled["mean_raw_predicted_harm"] == 0
    assert uncontrolled["mean_predicted_harm"] == pytest.approx(.05)
    assert not uncontrolled["predicted_constraints_satisfied"]


def test_invalid_shapes_scores_edges_and_zero_budget_fail_closed():
    p = problem()
    p.expected_gain[0] = np.nan
    with pytest.raises(ValueError):
        select_interventions(p, mode="joint")
    p = problem()
    p.edges[:] = [1, 0]
    with pytest.raises(ValueError):
        select_interventions(p, mode="joint")
    p = problem()
    p.max_interventions = 0
    assert not select_interventions(p, mode="joint")["switch"].any()


def test_edges_use_current_positions_and_proxy_is_relative_to_floor():
    current = np.array([[0., 0.], [1., 0.], [3., 0.]])
    np.testing.assert_array_equal(past_proximity_edges(current, radius=1.5), [[0, 1]])
    floor = np.repeat(current[:, None], 2, axis=1)
    edges = past_proximity_edges(current, radius=4)
    proxy = proximity_cost_table(floor, floor, edges, distance_threshold=2.)
    assert not proxy.any()


def test_pair_proxy_is_invariant_to_shared_rotation_translation_and_scale():
    rng = np.random.default_rng(31)
    floor, candidate = rng.normal(size=(2, 4, 8, 2))
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    rotate = np.array([[.6, -.8], [.8, .6]])
    before = proximity_cost_table(floor, candidate, edges, distance_threshold=2.)
    after = proximity_cost_table(floor @ rotate * 3 + 50, candidate @ rotate * 3 + 50,
                                 edges, distance_threshold=6.)
    np.testing.assert_allclose(before, after, atol=1e-14)


def test_empty_edge_set_does_not_change_cost_aware_choice():
    p = problem()
    p.edges = np.empty((0, 2), dtype=int)
    p.pair_cost = np.empty((0, 2, 2))
    result = select_interventions(p, mode="joint")
    np.testing.assert_array_equal(result["switch"], [True, False])
    assert result["mean_pair_proxy"] == 0


def test_labels_use_common_validity_masks_and_missing_endpoints_stay_missing():
    floor = np.zeros((2, 3, 2))
    candidate = np.ones((2, 3, 2))
    target = candidate.copy()
    target[1, -1] = np.nan
    mask = np.array([[True, True, True], [True, True, False]])
    y = realized_relative_costs(floor, candidate, target, mask, np.array([1., 2.]))
    np.testing.assert_allclose(y["gain_ade"], [np.sqrt(2), np.sqrt(2)/2])
    assert not y["harm_ade"].any()
    assert np.isnan(y["gain_fde"][1]) and not y["fde_valid"][1]
    empty = realized_relative_costs(floor, candidate, target, np.zeros_like(mask), np.ones(2))
    assert np.isnan(empty["gain_ade"]).all() and not empty["ade_valid"].any()


def test_cluster_calibration_rejects_duplicate_and_fitted_scene_ids():
    kwargs = dict(losses=np.zeros((2, 2, 1)), cluster_ids=["a", "a"],
                  fitted_cluster_ids=["train"], policy_ids=["p", "q"],
                  lower=np.array([0.]), upper=np.array([1.]), tolerance=np.array([.2]), delta=.05)
    with pytest.raises(ValueError, match="unique"):
        screen_cluster_risks(**kwargs)
    kwargs["cluster_ids"] = ["a", "train"]
    with pytest.raises(ValueError, match="fitted"):
        screen_cluster_risks(**kwargs)


def test_cluster_calibration_can_be_vacuous_and_never_invents_independence():
    kwargs = dict(losses=np.zeros((6, 2, 1)), cluster_ids=list("abcdef"),
                  fitted_cluster_ids=["train"], policy_ids=["p", "q"],
                  lower=np.array([0.]), upper=np.array([1.]), tolerance=np.array([.02]), delta=.05)
    result = screen_cluster_risks(**kwargs)
    assert not result["accepted"].any()
    assert not result["independence_verified"]
    kwargs["losses"][:] = -1
    with pytest.raises(ValueError, match="bounds"):
        screen_cluster_risks(**kwargs)
