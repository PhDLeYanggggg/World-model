import numpy as np
import pytest
from src.evaluation.m3w_record_risk_feasibility import witness


def test_non_easy_reference_cannot_subsidize_easy_risk():
    selected,rows = witness([10.,10000.],[10.3,0.],[1.,10000.],np.ones(2,bool),['a','a'],easy_cut=2.)
    assert not selected.any()
    assert rows['a']['reason'] == 'every_legal_easy_harm_ratio_exceeds_original_budget'


def test_exact_budget_and_beneficial_easy_switch_have_witnesses():
    selected,rows = witness([100.,10.],[102.,8.],[1.,1.],np.ones(2,bool),['a','b'],easy_cut=2.)
    assert selected.tolist() == [True,True]
    assert all(r['binary_fixed_proposal_finite_risk_feasible'] for r in rows.values())


def test_unknown_zero_reference_and_alias_do_not_create_support():
    selected,rows = witness([np.nan,0.,10.],[np.nan,0.,9.],[np.nan,1.,1.],
        np.array([True,True,False]),['unknown','zero','alias'],easy_cut=2.)
    assert not selected.any()
    assert rows['unknown']['unknown_rows'] == 1
    assert rows['zero']['reason'] == 'no_known_positive_easy_reference'
    assert rows['alias']['reason'] == 'no_legal_positive_easy_reference'
    assert rows['alias']['minimum_legal_easy_harm_ratio'] is None


def test_one_safe_easy_switch_per_record_is_sufficient_for_all_original_ratios():
    r=np.array([10.,20.,5.,50.]); p=np.array([10.1,100.,4.,300.]); cv=np.ones(4)
    selected,rows = witness(r,p,cv,np.ones(4,bool),['a','a','b','b'],easy_cut=2.)
    assert selected.tolist() == [True,False,True,False]
    for rec in ('a','b'):
        at=np.array(['a','a','b','b']) == rec
        use=selected & at
        harm=np.maximum(p[use]-r[use],0).sum()
        assert harm/r[use].sum() <= .02
        assert (p[use]-r[use]).sum()/r[at].sum() <= .02


def test_missing_costs_cannot_be_disguised_as_known_zero():
    with pytest.raises(ValueError,match='identical'):
        witness([np.nan],[0.],[np.nan],np.ones(1,bool),['a'],easy_cut=2.)
