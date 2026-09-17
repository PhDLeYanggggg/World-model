from copy import deepcopy

import pytest

from scripts.compare_m3w_residual_parameterizations import compare
from test_m3w_public_predictor_comparison import complete


def report():
    r = complete()
    r['seeds'] = [17, 29, 43]
    original_fits, original_rows = r['fits'], r['comparisons']
    r['fits'], r['comparisons'] = [], []
    for seed in r['seeds']:
        r['fits'].extend({**deepcopy(x), 'seed': seed} for x in original_fits)
        r['comparisons'].extend({**deepcopy(x), 'seed': seed} for x in original_rows)
        r['baseline_candidate_oracle_diagnostic'][str(seed)] = deepcopy(r['baseline_candidate_oracle_diagnostic']['17'])
        r['selection_by_seed'][str(seed)] = deepcopy(r['selection_by_seed']['17'])
    return r


def test_negative_gains_and_floor_retained_with_cached_reference():
    a, b, old = report(), report(), report()
    old['protocol_sha256'] = 'historical'
    result = compare(a, b, old)
    assert len(result['paired_seeds']) == 3
    assert all(r['bounded_selection']['arm'] == 'floor' for r in result['paired_seeds'])
    assert result['seed_descriptive_statistics']['motion_bounded']['gain_mean_pct'] == -5
    assert result['seed_descriptive_statistics']['v6_absolute_reference']['result_source'].startswith('cached_verified')


@pytest.mark.parametrize('change', ['seed', 'budget', 'duplicate', 'protocol', 'baseline', 'support'])
def test_incomplete_or_unmatched_ablation_rejected(change):
    a, b = report(), report()
    if change == 'seed':
        b['seeds'].pop()
    elif change == 'budget':
        b['fits'][0]['steps'] = 100
    elif change == 'duplicate':
        b['fits'].append(deepcopy(b['fits'][0]))
    elif change == 'protocol':
        b['protocol_sha256'] = 'changed'
    elif change == 'baseline':
        b['baselines']['site']['rows'] = 100
    else:
        b['comparisons'][0]['complete_queries'] += 1
    with pytest.raises(ValueError):
        compare(a, b)
