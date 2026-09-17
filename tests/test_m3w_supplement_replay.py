from copy import deepcopy

import pytest

from scripts.audit_m3w_supplement_replay import compare_rows


def row():
    return {'frame_id': 10, 'scale': 1., 'baseline_ade': 1., 'arms': {
        k: {'ade': 1., 'fde': None, 'switch': k == 'uncontrolled'}
        for k in ('floor', 'uncontrolled', 'independent', 'scene_uniform', 'joint')}}


def test_added_controls_do_not_change_fixed_replay():
    a = row()
    b = deepcopy(a)
    b['arms']['joint_exact_count'] = {'ade': .5}
    result = compare_rows([a], [b])
    assert result['fixed_forecast_scoring_identical']
    assert result['ordinary_control_rows_different'] == 0


def test_solver_decision_difference_is_reported_not_silently_hidden():
    a = row()
    b = deepcopy(a)
    b['arms']['joint']['switch'] = True
    result = compare_rows([a], [b])
    assert result['fixed_forecast_scoring_identical']
    assert result['ordinary_control_rows_different'] == 1


def test_changed_forecast_error_does_not_pass():
    a = row()
    b = deepcopy(a)
    b['arms']['uncontrolled']['ade'] += .125
    result = compare_rows([a], [b])
    assert not result['fixed_forecast_scoring_identical']
    assert result['max_forecast_error_absolute_difference'] == .125


def test_count_control_identity_difference_is_not_hidden_by_equal_error():
    a = row()
    b = deepcopy(a)
    b['arms']['independent_count_reference'] = {'switch': False, 'ade': 1.}
    b['arms']['joint_exact_count'] = {'switch': True, 'ade': 1.}
    result = compare_rows([a], [b])
    assert result['fixed_forecast_scoring_identical']
    assert result['count_control_queries_available'] == 1
    assert result['count_control_switch_disagreements'] == 1


@pytest.mark.parametrize('field,value', [('frame_id', 20), ('scale', 2.), ('baseline_ade', None)])
def test_changed_population_or_labels_rejected(field, value):
    a = row()
    b = deepcopy(a)
    b[field] = value
    with pytest.raises(ValueError, match='identity, scale or labels'):
        compare_rows([a], [b])
