from copy import deepcopy
import numpy as np
import pytest
from src.world_model.m3w_eqmotion_cost_data import assemble_view, validate_training_labels


def example():
    sites = np.repeat(list('abcd'), 3)
    groups = []
    for inner in 'bcd':
        ids = np.flatnonzero(np.isin(sites, ['a', inner]))
        labels = dict(benefit=ids.astype(float), harm=np.zeros(len(ids)),
            baseline_ade=np.ones(len(ids)), complete_future=np.ones(len(ids), bool))
        groups.append(dict(inner_site=inner, ids=ids,
            prediction=np.broadcast_to(ids[:, None, None], (len(ids), 12, 2)).astype(np.float32).copy(),
            labels=labels))
    return sites, groups


def test_pair_outer_rows_are_removed_and_targets_stay_separate():
    sites, groups = example()
    x, y = assemble_view(sites, 'a', groups)
    assert set(x) == {'ids', 'prediction'}
    np.testing.assert_array_equal(x['ids'], np.arange(3, 12))
    np.testing.assert_array_equal(x['prediction'][:, 0, 0], x['ids'])
    np.testing.assert_array_equal(y['benefit'], x['ids'])
    poisoned = deepcopy(groups)
    for g in poisoned:
        for field in ('benefit', 'harm', 'baseline_ade'):
            g['labels'][field][:] = np.nan
    px, _ = assemble_view(sites, 'a', poisoned)
    np.testing.assert_array_equal(px['prediction'], x['prediction'])


@pytest.mark.parametrize('kind', ['missing', 'duplicate_group', 'duplicate_id', 'outer_only', 'nonfinite'])
def test_invalid_coverage_and_prediction_rejected(kind):
    sites, groups = example()
    if kind == 'missing': groups.pop()
    elif kind == 'duplicate_group': groups.append(deepcopy(groups[0]))
    elif kind == 'duplicate_id': groups[0]['ids'][1] = groups[0]['ids'][0]
    elif kind == 'outer_only': groups[0]['inner_site'] = 'a'
    else: groups[0]['prediction'][0, 0, 0] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        assemble_view(sites, 'a', groups)


def test_unknown_costs_are_not_converted_to_zero():
    sites, groups = example()
    for field in ('benefit', 'harm', 'baseline_ade'):
        groups[0]['labels'][field][3] = np.nan
    groups[0]['labels']['complete_future'][3] = False
    _, y = assemble_view(sites, 'a', groups)
    assert np.isnan(y['harm'][0]) and not y['complete_future'][0]
    validate_training_labels(y, y['complete_future'])


@pytest.mark.parametrize('kind', ['complete_mismatch', 'negative', 'infinite', 'unpaired_nan', 'both_costs'])
def test_supervision_support_rejected_without_using_it_as_input(kind):
    sites, groups = example()
    _, y = assemble_view(sites, 'a', groups)
    complete = y['complete_future'].copy()
    if kind == 'complete_mismatch': complete[0] = False
    elif kind == 'negative': y['harm'][0] = -1
    elif kind == 'infinite': y['harm'][0] = np.inf
    elif kind == 'unpaired_nan': y['harm'][0] = np.nan
    else: y['harm'][0] = 1
    with pytest.raises((ValueError, AssertionError)):
        validate_training_labels(y, complete)
