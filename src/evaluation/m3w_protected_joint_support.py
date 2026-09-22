"""Outcome-free non-additive support audit at the inherited half-count budget."""
import itertools
import math

import numpy as np

from src.world_model.m3w_interaction_controls import decompose_pair_objective
from src.world_model.m3w_native_joint_controls import rank_reference


def audit_problem(problem, agent_ids, *, enumeration_cap=200000):
    """Inspect opportunity, not realized accuracy or a deployable policy.

    The feasible set is anchored to the old top-gain half-count rule. Exact
    enumeration is only needed when two eligible actions have a product term.
    Oversized sets stay explicitly unverified rather than being subsampled.
    """
    p = problem
    p.validate()
    if not isinstance(enumeration_cap, int) or enumeration_cap < 1:
        raise ValueError('Positive integer enumeration cap required')
    ids = np.asarray(agent_ids)
    reference = rank_reference(p, ids, .5)
    k, n = int(reference.sum()), len(reference)
    parts = decompose_pair_objective(p)
    risk = parts['original_coherent_harm']
    budget = float(np.mean(reference*risk))
    pool = np.flatnonzero(p.supported)
    pool = pool[np.argsort(ids[pool], kind='stable')]
    active = p.supported[p.edges].all(1) & (parts['product_objective'] != 0)
    combinations = math.comb(len(pool), k)
    result = dict(pool=len(pool), count=k, combinations=combinations,
        predicted_harm_budget=budget, nonadditive_edges=int(active.sum()),
        eligible_product_absolute_bound=float(np.abs(parts['product_objective'][active]).sum()),
        future_labels_used=False, exact_enumeration=False, feasible=None,
        changed_identities=None, objective_advantage=None, meaningful_advantage=None)
    if k < 2 or not active.any():
        return dict(result, status='structural_null', meaningful_advantage=False,
                    objective_advantage=0., changed_identities=0)
    if combinations > enumeration_cap:
        return dict(result, status='enumeration_cap_blocker')
    values, subsets = [], []
    for chosen in itertools.combinations(pool, k):
        bits = np.zeros(n, bool)
        bits[list(chosen)] = True
        if float(np.mean(bits*risk)) > budget+1e-10:
            continue
        unary = float(parts['unary_objective']@bits)
        product = float(parts['product_objective']@(bits[p.edges[:, 0]] & bits[p.edges[:, 1]]))
        direct_pair = p.pair_cost[np.arange(len(p.edges)), bits[p.edges[:, 0]].astype(int),
                                 bits[p.edges[:, 1]].astype(int)]
        direct = -float(np.mean(bits*p.expected_gain))+p.pair_weight*float(direct_pair.mean())
        np.testing.assert_allclose(unary+product, direct, atol=1e-12, rtol=1e-12)
        values.append((unary, direct, product))
        subsets.append(chosen)
    if not values:
        raise ArithmeticError('The reference must be feasible in its own harm budget')
    values = np.asarray(values)
    ui, ji = np.argmin(values[:, 0]), np.argmin(values[:, 1])
    advantage = float(values[ui, 1]-values[ji, 1])
    tolerance = 1e-9*(1+abs(values[ui, 1])+abs(values[ji, 1]))
    if advantage < -tolerance:
        raise ArithmeticError('Exhaustive joint minimum cannot exceed unary choice')
    order = np.sort(values[:, 0])
    gap = None if len(order) < 2 else float(order[1]-order[0])
    product_range = float(np.ptp(values[:, 2]))
    return dict(result, status='exhaustively_checked', exact_enumeration=True,
        feasible=len(values), changed_identities=len(set(subsets[ui]) ^ set(subsets[ji])),
        objective_advantage=advantage, meaningful_advantage=bool(advantage > tolerance),
        objective_tolerance=float(tolerance), product_range=product_range,
        unary_runner_up_gap=gap,
        product_range_below_unary_gap=None if gap is None else bool(product_range < gap))
