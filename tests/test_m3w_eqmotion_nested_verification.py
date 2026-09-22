from copy import deepcopy
import pytest
from scripts.verify_m3w_eqmotion_nested import audit_lineage


def example():
    return dict(seed=17, family='eqmotion_fixed_head', excluded_sites=['a', 'b'],
        training_sites=['c', 'd'], preprocessing_fit_sites=['c', 'd'],
        parents=[], initialization='random_seed', checkpoint_selection_sites=[],
        calibration_sites=[], research_design_exposed_sites=list('abcd'))


def test_independent_check_requires_both_exclusions():
    audit_lineage(example(), 'a', 'b', 17, list('abcd'))


@pytest.mark.parametrize('field,value', [
    ('training_sites', ['b', 'c', 'd']), ('preprocessing_fit_sites', ['a', 'c', 'd']),
    ('parents', ['leaked_teacher']), ('seed', 29), ('excluded_sites', ['a']),
    ('checkpoint_selection_sites', ['a']), ('calibration_sites', ['b']),
    ('family', 'transformer'), ('research_design_exposed_sites', [])])
def test_independent_check_rejects_hidden_exposure(field, value):
    record = deepcopy(example()); record[field] = value
    with pytest.raises(ValueError):
        audit_lineage(record, 'a', 'b', 17, list('abcd'))
