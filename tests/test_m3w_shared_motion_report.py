import pytest

from scripts.report_m3w_moving_shared_motion_cost import interval, render


def test_undefined_control_is_not_zero_or_safety_pass():
    assert interval(dict(point=None, CI95=None)) == 'not estimable'
    assert interval(dict(point=0., CI95=[0., 0.])) == '+0.0000 [+0.0000, +0.0000]'


def test_report_keeps_primary_and_empty_utility_boundaries():
    empty = dict(point=None, CI95=None)
    summary = dict(original24group_primary='not_estimable_not_passed',
        matched_utility_is_not_safety_gate=True,
        count_matching_within_arm_not_between_feature_arms=True,
        feature_effect_uses_full_original_population=True,
        paired_shared_minus_local={}, safety={},
        within_arm_nonempty_matched_utility=dict(local_history=dict(
            known_pp=empty, completion_lower_pp=empty)),
        matched_utility_support=dict(local_history=dict(nonempty_query_views=0,
            empty_query_views=5, nonestimable_source_views=1)))
    training = dict(complete_heads=36, optimizer_updates=72000)
    execution = dict(seconds=1., peak_RSS_bytes=1)
    text = render(summary, training, execution)
    assert '0/5 | 1 | not estimable | not estimable' in text
    assert 'original 24-group primary is not estimable and has not passed' in text
    assert 'Deployment unchanged' in text
    summary['matched_utility_is_not_safety_gate'] = False
    with pytest.raises(ValueError, match='utility-versus-safety'):
        render(summary, training, execution)
