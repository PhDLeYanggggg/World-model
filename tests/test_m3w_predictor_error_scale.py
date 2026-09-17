import pytest

from scripts.analyze_m3w_predictor_error_scale import decomposition


def row(scene, scale, b, c):
    return {'physical_scene': scene, 'recording_id': scene, 'scale': scale,
            'baseline_ade': b, 'arms': {'uncontrolled': {'ade': c}}}


def test_error_decomposition_preserves_signed_difference_and_physical_weight():
    result = decomposition([row('a', .001, 1, 3), row('a', 1, 4, 2), row('b', 1, 2, 4)])
    assert result['net_excess'] == pytest.approx(1.)
    assert sum(r['net_excess_contribution_to_primary_ade'] for r in result['bins'].values()) == pytest.approx(result['net_excess'])
    assert result['bins']['numerical_scale_floor']['fraction_of_all_positive_harm'] == pytest.approx(1/3)
    assert len(result['per_recording_native_only']) == 2


def test_missing_labels_not_given_zero_error_and_zero_harm_ratio_is_undefined():
    result = decomposition([row('a', 1, 2, 1), row('a', .001, None, None)])
    assert result['complete_agent_queries'] == 1
    assert result['bins']['numerical_scale_floor']['fraction_of_all_positive_harm'] is None
    with pytest.raises(ValueError, match='Complete-path'):
        decomposition([row('a', 1, None, None)])
