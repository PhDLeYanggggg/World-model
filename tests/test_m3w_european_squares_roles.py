from collections import Counter
import pytest
from src.evaluation.m3w_european_squares_roles import assign_roles, require_source_training


def test_assignment_balanced_deterministic_not_insertion_order():
    support = {f'g{i:03d}': i*123 for i in range(36)}
    result = assign_roles(support)
    assert result == assign_roles(dict(reversed(list(support.items()))))
    assert Counter(r['role'] for r in result.values()) == {
        'source_training': 12, 'risk_calibration_reserved': 12,
        'model_selection_reserved': 6, 'confirmation_reserved': 6}
    for block in range(6):
        assert Counter(r['role'] for r in result.values() if r['support_block'] == block) == {
            'source_training': 2, 'risk_calibration_reserved': 2,
            'model_selection_reserved': 1, 'confirmation_reserved': 1}


def test_bad_group_count_and_support_fail_closed():
    with pytest.raises(ValueError):
        assign_roles({'g': 1})
    with pytest.raises(ValueError):
        assign_roles({str(i): -i for i in range(36)})


def manifest(role='source_training', training=True):
    return dict(status='restricted_source_training_admitted_reserved_roles_closed',
                recordings=[dict(source_member='a.csv', role=role, training_access=training)])


@pytest.mark.parametrize('role', ['confirmation_reserved', 'risk_calibration_reserved', 'model_selection_reserved'])
def test_reserved_roles_never_train_even_if_access_flag_wrong(role):
    with pytest.raises(PermissionError):
        require_source_training(manifest(role), 'a.csv')


def test_training_requires_both_role_and_access():
    assert require_source_training(manifest(), 'a.csv')['training_access']
    with pytest.raises(PermissionError):
        require_source_training(manifest(training=False), 'a.csv')
    with pytest.raises(ValueError):
        require_source_training(manifest(), 'absent.csv')
    m = manifest()
    m['recordings'] *= 2
    with pytest.raises(ValueError):
        require_source_training(m, 'a.csv')
    with pytest.raises(ValueError):
        require_source_training({'status': 'unassigned'}, 'a.csv')
