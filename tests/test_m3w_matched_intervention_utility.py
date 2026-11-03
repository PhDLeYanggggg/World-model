import numpy as np
import pytest

from src.evaluation.m3w_matched_intervention_utility import matched_utility


def example():
    floor = np.zeros((4, 12, 2))
    floor[..., 0] = 10
    first, second = floor.copy(), floor.copy()
    first[2, :, 0] = 12
    second[3, :, 0] = 0
    return dict(independent_path=first, joint_path=second, floor=floor,
        target=np.zeros_like(floor), valid=np.ones((4, 12), bool),
        independent_choice=np.array([-1, -1, 6, -1]), joint_choice=np.array([-1, -1, -1, 6]),
        groups=[np.array([0, 1]), np.array([2, 3])])


def test_nonempty_population_is_causal_and_never_a_safety_gate():
    data = example()
    result = matched_utility(**data)
    assert result['nonempty_queries'] == 1
    assert result['empty_queries'] == 1
    assert result['eligible_rows'] == 2
    assert result['matched_interventions_each_policy'] == 1
    assert result['contrast']['known_pp'] == 60
    assert result['safety_gate_usage'] is False
    data['valid'][:] = False
    changed = matched_utility(**data)
    assert changed['eligible_rows'] == result['eligible_rows']
    assert changed['nonempty_queries'] == result['nonempty_queries']
    assert changed['contrast']['known_pp'] is None


def test_empty_matched_control_is_not_a_zero_effect_or_safety_pass():
    data = example()
    data.update(independent_path=data['floor'].copy(), joint_path=data['floor'].copy(),
        independent_choice=np.full(4, -1), joint_choice=np.full(4, -1))
    result = matched_utility(**data)
    assert result['status'] == 'not_estimable_empty_matched_control'
    assert result['contrast']['known_pp'] is None
    assert result['contrast']['completion_lower_pp'] is None
    assert result['safety_gate_usage'] is False
    assert result['full_strategy_risk_population_unchanged']


def test_mismatched_count_or_nonpartition_cannot_escape():
    data = example()
    data['joint_choice'][2] = 6
    with pytest.raises(ValueError, match='every query'):
        matched_utility(**data)
    data = example()
    data['groups'] = [np.array([0, 1]), np.array([1, 2, 3])]
    with pytest.raises(ValueError, match='exactly one query'):
        matched_utility(**data)
