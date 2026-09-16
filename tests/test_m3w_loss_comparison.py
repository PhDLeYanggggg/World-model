from copy import deepcopy

import pytest

from scripts.compare_m3w_8to12_losses import paired_rows


def example():
    return {'task': {'horizon': 12}, 'seeds': [17], 'baselines': {'site': {'rows': 5}},
            'comparisons': [{'seed': 17, 'arm': 'uncontrolled', 'ade': 2., 'fde': 3.,
                             'complete_queries': 5, 'fde_endpoint_label_queries': 6,
                             'past_supported_queries': 7, 'scene_count': 1}],
            'baseline_candidate_oracle_diagnostic': {'17': {}},
            'selection_by_seed': {'17': {'selected': {'arm': 'floor'}}}}


def test_paired_reduction_does_not_upgrade_floor():
    a, b = example(), example()
    b['comparisons'][0]['ade'] = 1.5
    row = paired_rows(a, b)[0]
    assert row['candidate_ade_reduction_vs_mse_pct'] == 25
    assert row['robust_selection']['arm'] == 'floor'


@pytest.mark.parametrize('mutation', ['task', 'seeds', 'baselines', 'support', 'predictions'])
def test_unmatched_experiments_rejected(mutation):
    a, b = example(), example()
    if mutation == 'task':
        b['task']['horizon'] = 50
    elif mutation == 'seeds':
        b['seeds'] = [29]
    elif mutation == 'baselines':
        b['baselines']['site']['rows'] = 6
    elif mutation == 'support':
        b['comparisons'][0]['complete_queries'] = 6
    else:
        extra = deepcopy(b['comparisons'][0])
        extra['ade'] += 1
        b['comparisons'].append(extra)
    with pytest.raises(ValueError):
        paired_rows(a, b)
