import pytest
from scripts.report_m3w_query_unit_critic import common_curve


def test_changed_training_units_are_compared_in_original_units():
    old = [dict(step=0, monitor=dict(pointwise_risk=.2, selected_query_score=.01))]
    new = [dict(step=0, monitor=dict(pointwise_risk=.2, selected_query_score=20.,
        selected_query_score_original_units=.01))]
    left, right = common_curve(old, 'old'), common_curve(new, 'new')
    assert left == right
    assert left['original_objective'] == pytest.approx([.105])
    assert left['pointwise'] == [.2] and left['query'] == [.01]


def test_missing_original_units_refuses_instead_of_mixing_losses():
    trace = [dict(step=0, monitor=dict(pointwise_risk=.2, selected_query_score=20.))]
    with pytest.raises(ValueError): common_curve(trace, 'new')
    with pytest.raises(ValueError): common_curve(trace, 'unsupported')


def test_trace_steps_must_be_complete_unique_and_ordered():
    row = dict(step=0, monitor=dict(pointwise_risk=.2, selected_query_score=.01))
    with pytest.raises(ValueError): common_curve([row, row], 'old')
