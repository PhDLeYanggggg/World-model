from copy import deepcopy

import pytest

from reproducibility.m3w_integrated_reader.verify import compare, evidence


def fixture():
    values = {
        'uniform': {'expected_passes_low': 49.29},
        'count': {'views': 606, 'edits': 780, 'full_grid': 210},
        'listwise': {'uniform': 46.73},
        'relative': {'support': {'pair_steps': 5842092}},
        'masking': {'support': {'queries': 3711}},
    }
    keys = ('opendd_fixed_spend_uniform', 'opendd_full_grid_count_motion',
            'opendd_listwise_full_grid', 'opendd_backbone_relative', 'opendd_pair_label_stress')
    tree = {}
    for k, v in zip(keys, values.values(), strict=True):
        tree = {k: deepcopy(v), 'inherited': tree}
    return values, tree


def test_reader_reconciles_nested_evidence():
    values, tree = fixture()
    assert compare(values, tree) == dict(views=606, edits=780, full_grid=210,
                                         uniform_expected=49.29, complete_queries=3711,
                                         relative_pair_steps=5842092)
    with pytest.raises(ValueError, match='evidence missing'):
        evidence(tree, 'not_measured')


def test_reader_rejects_changed_table_even_if_archive_is_rehashed():
    values, tree = fixture()
    values['count']['full_grid'] += 1
    with pytest.raises(ValueError, match='tables do not match'):
        compare(values, tree)
