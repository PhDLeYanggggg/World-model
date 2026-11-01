import copy
import numpy as np
import pytest
from scripts.verify_m3w_reference_refit_coherence import coherence


def test_diagnostic_detects_incoherence_without_altering_any_prediction():
    p=np.ones((2,7,5));mask=np.ones((2,7),bool);mask[:,-1]=False
    p[0,0,0]=3.;p[1,0,2]=2.;p[1,0,3]=3.;p[1,0,4]=2.
    p[:,-1,0]=1e12;before=p.copy();v=coherence(p,mask,np.ones(2))
    np.testing.assert_array_equal(p,before)
    assert v['legal_action_views']==12
    assert v['violations']['benefit_exceeds_reference']['above1e_12_normalized_action_views']==1
    assert v['violations']['negative_implied_candidate_ADE']['above1e_12_normalized_action_views']==1
    assert v['violations']['easy_reference_exceeds_reference']['above1e_12_normalized_action_views']==1
    assert v['violations']['easy_harm_exceeds_harm']['above1e_12_normalized_action_views']==1
    assert v['reference_spread']['reference']['rows_with_spread_above1e_12']==1
    assert v['inference_only_no_targets'] and v['diagnostic_not_a_new_gate']


def test_empty_legal_support_is_not_observed_zero_and_bad_inputs_rejected():
    p=np.ones((2,7,5));mask=np.zeros((2,7),bool);v=coherence(p,mask,np.ones(2))
    assert v['violations']['negative_implied_candidate_ADE']['max_positive_excess_over_causal_scale'] is None
    assert v['reference_spread']['reference']['mean_spread_over_causal_scale'] is None
    with pytest.raises(ValueError):coherence(p,mask,np.zeros(2))
    bad=copy.deepcopy(p);bad[0,0,0]=np.nan
    with pytest.raises(ValueError):coherence(bad,mask,np.ones(2))
