import itertools
import math
import numpy as np
import pytest
from src.world_model.m3w_risk_subsidy_controls import MODES, coefficients, direct_risk, select


def test_nonselected_denominator_can_fund_single_intervention():
    g, q, r, ok = np.array([2., 0.]), np.array([.03, 0.]), np.array([1., 10.]), np.array([True, False])
    assert select(g, q, r, ok, 'net_population')[0][0]
    assert not select(g, q, r, ok, 'net_selected')[0].any()


def test_negative_credit_cannot_fund_other_agent_in_clipped_arm():
    g, q, r, ok = np.array([1., 2.]), np.array([-5., 3.]), np.ones(2), np.ones(2, bool)
    assert select(g, q, r, ok, 'net_population')[0].all()
    np.testing.assert_array_equal(select(g, q, r, ok, 'net_clipped_population')[0], [True, False])
    assert select(g, q, r, ok, 'net_selected')[0].all()


@pytest.mark.parametrize('mode', MODES)
def test_small_queries_against_exhaustive_original_inequality(mode):
    rng = np.random.default_rng(94287)
    for _ in range(20):
        g, q, r = rng.uniform(.1, 2, 6), rng.uniform(-.03, .08, 6), rng.uniform(0, 2, 6)
        ok = rng.random(6) > .2
        possible = []
        for v in itertools.product([False, True], repeat=6):
            b = np.array(v); risk, budget = direct_risk(q, r, b, mode)
            if not (b & ~ok).any() and risk <= budget: possible.append(b)
        for count in (None, min(int(b.sum()) for b in possible), max(int(b.sum()) for b in possible)):
            b, report = select(g, q, r, ok, mode, count=count)
            assert report['direct_constraint_pass'] and not (b & ~ok).any()
            if report['optimal']:
                feasible = [v for v in possible if count is None or v.sum() == count]
                assert math.fsum(g[b]) == pytest.approx(max(math.fsum(g[v]) for v in feasible), abs=1e-8)
            elif report['failed_closed']: assert not b.any()


def test_selected_denominator_risk_may_be_negative_without_negative_net():
    q, budget = coefficients(np.array([.01, .5]), np.array([1., 1.]), 'net_clipped_selected')
    np.testing.assert_allclose(q, [-.01, .48]); assert budget == 0


@pytest.mark.parametrize('net,den,mode,rho', [([np.nan], [1], 'net_selected', .02),
    ([0], [-1], 'net_selected', .02), ([0], [1], 'other', .02), ([0], [1], 'net_selected', .03)])
def test_rejects_invalid_or_changed_scientific_contract(net, den, mode, rho):
    with pytest.raises(ValueError): coefficients(net, den, mode, rho)


def test_query_count_cannot_be_forced_when_infeasible():
    bits, report = select(np.ones(2), np.ones(2), np.ones(2), np.ones(2, bool),
        'net_clipped_selected', count=1)
    assert not bits.any() and not report['optimal'] and not report['exact_count_pass']


def test_canonical_boundary_not_lost_to_transformed_rounding():
    r = np.array([.1, .2]); q = np.array([.02*math.fsum(r), 0.])
    b, info = select(np.array([2., 1.]), q, r, np.ones(2,bool), 'net_clipped_selected')
    assert b.all() and info['canonical_optimal_verified']


def test_large_numerical_solution_never_claims_canonical_optimality():
    g = np.arange(1,13, dtype=float); q=np.full(12,.021);r=np.ones(12);q[0]=0
    b, info = select(g,q,r,np.ones(12,bool),'net_clipped_selected',count=2)
    assert info['direct_constraint_pass']
    assert not info['optimal'] and not info['canonical_optimal_verified']
    if not info['failed_closed']: assert b.sum()==2
