import numpy as np
import pytest

from src.evaluation.m3w_duration_support import support_partitions


def test_disjoint_label_support_preserves_every_selected_cost_mass():
    p = np.arange(20, dtype=float).reshape(4, 5) / 100
    y = p * 3
    mask = np.zeros((4, 12), bool)
    mask[0] = True
    mask[1, :5] = True
    mask[3] = True
    selected = np.array([True, True, True, False])
    out = support_partitions(p, y, selected, mask)
    assert {k:v['rows'] for k,v in out['parts'].items()} == dict(complete=1, partial=1, unknown=1)
    assert out['whole']['EH']['target_mass'] == y[:3, 4].sum()
    assert out['unknown_share_of_target_EH'] == y[2, 4]/y[:3, 4].sum()
    assert out['observed_only_risks']['target_easy'] == y[:2, 4].sum()/y[:2, 3].sum()


def test_empty_support_does_not_become_zero_risk():
    out = support_partitions(np.zeros((2, 5)), np.zeros((2, 5)),
                             np.zeros(2, bool), np.zeros((2, 12), bool))
    assert out['unknown_share_of_target_EH'] is None
    assert out['whole_risks']['target_easy'] is None
    assert out['observed_only_risks']['target_easy'] is None


def test_label_support_cannot_be_implicit_numeric_or_wrong_horizon():
    p = np.ones((2, 5))
    for mask in (np.ones((2, 12)), np.ones((2, 11), bool)):
        with pytest.raises(ValueError):
            support_partitions(p, p, np.ones(2, bool), mask)
