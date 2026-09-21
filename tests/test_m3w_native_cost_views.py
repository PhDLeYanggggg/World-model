import copy

import numpy as np
import pytest

from src.world_model.m3w_native_cost_views import assemble_training_view
from src.world_model.m3w_native_nested import cost_supervision


def groups():
    sites = np.repeat(['a', 'b', 'c', 'd'], 3)
    result = []
    for inner in 'bcd':
        ids = np.flatnonzero(np.isin(sites, ['a', inner]))
        p = np.full((len(ids), 12, 2), 2, np.float32)
        baseline = p*.5; target = p*.7; valid = np.ones((len(ids), 12), bool)
        valid[-1] = False; target[-1] = np.nan
        producer = dict(id='a'+inner, seed=17, excluded_sites=['a', inner],
            training_sites=sorted(set(sites)-{'a', inner}),
            preprocessing_fit_sites=sorted(set(sites)-{'a', inner}), parents=[],
            initialization='random_seed', checkpoint_selection_sites=[], calibration_sites=[],
            research_design_exposed_sites=['a', 'b', 'c', 'd'], objective='native_coordinate')
        labels = cost_supervision(p, baseline, target, valid, np.ones(len(ids)))
        result.append(dict(inner_site=inner, ids=ids, prediction=p, labels=labels, producer=producer))
    return sites, result


def test_physical_training_partition_excludes_outer_and_separates_labels():
    sites, g = groups()
    x, y = assemble_training_view('a', 17, sites, g)
    assert set(x) == {'ids', 'prediction'}
    np.testing.assert_array_equal(x['ids'], np.arange(3, 12))
    np.testing.assert_array_equal(x['ids'], y['ids'])
    assert np.isnan(y['gain']).sum() == 3


def test_outer_payload_poison_cannot_change_training_view():
    sites, g = groups(); first = assemble_training_view('a', 17, sites, g)
    for r in g:
        mask = sites[r['ids']] == 'a'
        r['prediction'][mask] = np.nan
        for k in r['labels']:
            r['labels'][k][mask] = False if r['labels'][k].dtype == bool else 999999
    second = assemble_training_view('a', 17, sites, g)
    for one, two in zip(first, second):
        for key in one:
            np.testing.assert_array_equal(one[key], two[key])


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'outer', 'ids', 'upstream', 'label', 'support', 'unknown_inf'])
def test_invalid_partition_cannot_be_used_for_cost_training(fault):
    sites, g = groups(); g = copy.deepcopy(g)
    if fault == 'missing':
        g.pop()
    elif fault == 'duplicate':
        g[1] = g[0]
    elif fault == 'outer':
        g[0]['inner_site'] = 'a'
    elif fault == 'ids':
        g[0]['ids'][0] = g[0]['ids'][1]
    elif fault == 'upstream':
        g[0]['producer']['preprocessing_fit_sites'].append('a')
    elif fault == 'support':
        g[0]['labels']['supported_steps'][3] = 99
    elif fault == 'unknown_inf':
        g[0]['labels']['gain'][-1] = np.inf
    else:
        g[0]['labels']['harm'][3] += 1
    with pytest.raises((ValueError, AssertionError)):
        assemble_training_view('a', 17, sites, g)
