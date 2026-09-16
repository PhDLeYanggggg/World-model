from copy import deepcopy

import pytest

from scripts.compare_m3w_public_predictors import compare
from test_m3w_loss_comparison import example


def complete():
    p = example()
    p.update({'protocol_sha256': 'same', 'independent_confirmation': False,
        'fits': [{'seed': 17, 'model': n, 'training_complete': True,
                  'steps': 1000 if n == 'neural_cost' else 10000}
                 for n in ('full', 'hold0', 'hold1', 'hold2', 'neural_cost')]})
    p['comparisons'][0]['all_improvement_pct'] = -5.
    p['baseline_candidate_oracle_diagnostic']['17']['per_recording_native_units'] = {}
    return p


def test_negative_forecast_and_floor_are_retained():
    a, b = complete(), complete()
    r = compare(a, b)
    assert r['seed_descriptive_statistics']['eqmotion_K1']['gain_mean_pct'] == -5
    assert r['paired_seeds'][0]['eqmotion_selection']['arm'] == 'floor'


@pytest.mark.parametrize('change', ['protocol', 'confirmation', 'partial', 'steps', 'missing', 'duplicate'])
def test_incomplete_or_incompatible_comparison_refused(change):
    a, b = complete(), complete()
    if change == 'protocol':
        b['protocol_sha256'] = 'other'
    elif change == 'confirmation':
        b['independent_confirmation'] = True
    elif change == 'partial':
        b['fits'][0]['training_complete'] = False
    elif change == 'steps':
        b['fits'][0]['steps'] = 100
    elif change == 'missing':
        b['fits'].pop()
    else:
        b['fits'].append(deepcopy(b['fits'][0]))
    with pytest.raises(ValueError):
        compare(a, b)
