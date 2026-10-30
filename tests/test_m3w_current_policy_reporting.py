import pytest
from scripts.report_m3w_current_policy_critic import comparable_curve


def test_only_fixed_original_set_probe_is_comparable_across_training_targets():
    fixed = [dict(step=0, monitor=dict(pointwise_risk=.2, selected_query_score_original_units=.01))]
    moving = [dict(step=0, monitor=dict(pointwise_risk=.2, selected_query_score_original_units=99.,
        frozen_selected_query_score_original_units=.01, complete_selected_queries=3, current_selected_rows=10))]
    left, right = comparable_curve(fixed, 'fixed'), comparable_curve(moving, 'current')
    assert left['frozen_probe'] == right['frozen_probe'] == [.01]
    assert left['moving_query_term'] is None
    assert right['moving_query_term'] == [99.]
    assert right['complete_selected_queries'] == [3] and right['selected_rows'] == [10]


def test_missing_original_set_probe_does_not_relabel_current_query_loss():
    trace = [dict(step=0, monitor=dict(pointwise_risk=.2, selected_query_score_original_units=99.))]
    with pytest.raises(ValueError): comparable_curve(trace, 'current')
    with pytest.raises(ValueError): comparable_curve(trace, 'unknown')
