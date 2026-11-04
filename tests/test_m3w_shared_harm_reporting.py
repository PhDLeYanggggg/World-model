import pytest
from scripts.report_m3w_moving_shared_harm_refinement import aggregate_costs,matching_and_features


def test_aggregate_uses_same_known_population_not_all_selected_predictions():
    cell=dict(selected=5,selected_known=3,realized_selected_known_moments=[2,3,10,4,1],
        predicted_on_selected_known_moments=[1,1,5,2,.5])
    result=aggregate_costs([cell,cell])
    assert result['selected']==10 and result['selected_known']==6
    assert result['moments']['H']['ratio']==3
    assert result['risk']['all']['known_pooled_risk']==pytest.approx(.3)


def test_no_known_support_is_not_safe_zero_ratio():
    cell=dict(selected=3,selected_known=0,realized_selected_known_moments=[0]*5,
        predicted_on_selected_known_moments=[0]*5)
    result=aggregate_costs([cell])
    assert result['moments']['EH']['ratio']is None
    assert result['risk']['easy']['known_pooled_risk']is None


def test_incomplete_real_pair_inventory_cannot_pass_reporting_gate():
    with pytest.raises(ValueError,match='All18fresh matched pairs'):
        matching_and_features([])
