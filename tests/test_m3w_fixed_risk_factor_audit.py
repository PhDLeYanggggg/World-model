import numpy as np
import pytest
from src.evaluation.m3w_fixed_risk_factor_audit import factors,aggregate


def record(p,pk,a,y,unknown=0):
    return dict(predicted_full_selected_moments=p,predicted_on_selected_known_moments=pk,
        realized_selected_known_moments=a,bounded_label_selected_moments=y,
        selected=2,selected_known=2-unknown,selected_unknown=unknown)


def test_known_reference_overestimate_is_separate_from_harm():
    p=[1,1,100,100,1];a=[1,1,10,10,1]
    out=factors(record(p,p,a,a))['easy']
    assert out['predicted_feasible_but_bounded_violating']
    assert out['components']['known_harm']==0
    assert out['components']['known_reference']==pytest.approx(1.8)
    assert out['minimal_positive_masks']==[2]


def test_unknown_bound_and_reference_credit_can_jointly_hide_failure():
    p=[1,1,200,200,1];pk=[1,1,100,100,1];a=pk;y=[1,3,100,100,3]
    out=factors(record(p,pk,a,y,unknown=1))['all']
    assert out['minimal_positive_masks']==[12]
    assert out['components']['unknown_bound_harm']==2
    assert out['components']['unknown_reference_credit']==2
    assert out['counterfactual_excess_by_mask'][15]==pytest.approx(out['completion_excess'])


def test_absent_support_never_becomes_safe_zero_ratio():
    p=[1,1,100,40,.1];a=[0]*5
    out=factors(record(p,p,a,a))
    assert out['easy']['known_risk']is None
    assert out['easy']['no_observed_selected_reference']
    assert out['easy']['predicted_positive_known_reference_absent']
    rows=[dict(site='s',factors=out)]
    assert aggregate(rows,'easy')['counts']['no_observed_selected_reference']==1


def test_empty_action_and_invalid_reference_contract_are_explicit():
    r=record([0]*5,[0]*5,[0]*5,[0]*5);r.update(selected=0,selected_known=0)
    out=factors(r)['easy'];assert not out['active']and out['known_risk']is None
    r['bounded_label_selected_moments'][3]=1
    with pytest.raises(ValueError):factors(r)


def test_algebra_holds_for_signed_components_and_all_masks():
    p=np.array([1,2,100,30,1.]);pk=p*.5;a=np.array([1,3,40,5,1.]);y=a.copy();y[[1,4]]+=4
    for cell in factors(record(p,pk,a,y,unknown=1)).values():
        assert cell['predicted_excess']+sum(cell['components'].values())==pytest.approx(cell['completion_excess'])
