import numpy as np
import pytest
from scripts.report_m3w_european_selection_readout import aggregate_seeds


def rows(values):
    return [dict(metadata=dict(seed=seed), views=dict(test=dict(ADE_vs_incumbent=dict(all=dict(
        expected_scenes=['a', 'b'], by_scene=dict(a=dict(gain_percent=a), b=dict(gain_percent=b)))))))
        for seed, (a, b) in zip([17, 29, 43], values)]


def test_all_seeds_averaged_before_source_bootstrap():
    r = aggregate_seeds(rows([(2, 4), (-2, -4), (0, 0)]), 'test')
    assert r['mean'] == 0 and r['ci'] == [0, 0] and r['localities'] == 2
    assert r['seeds'] == 3


def test_favorable_seed_cannot_be_selected():
    with pytest.raises(ValueError): aggregate_seeds(rows([(20, 40)]), 'test')


def test_undefined_support_not_dropped():
    r = aggregate_seeds(rows([(None, 4), (-2, -4), (0, 0)]), 'test')
    assert r['mean'] is None and r['ci'] is None
