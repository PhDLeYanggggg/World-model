import numpy as np
import pytest

from src.evaluation.m3w_completion_occlusion import corridor, pseudo_gaps, summary


def test_causal_center_does_not_use_hidden_coordinates():
    y = np.column_stack((np.arange(12, dtype=float), np.zeros(12)))
    seen = np.arange(12) < 8
    hidden = np.arange(8, 12)
    p, radius = corridor(y, seen, hidden, 'ols4')
    np.testing.assert_allclose(p, y[hidden])
    np.testing.assert_allclose(radius, [1., 2., 3., 4.])
    y[hidden] = np.nan
    q, r = corridor(y, seen, hidden, 'ols4')
    np.testing.assert_array_equal(p, q)
    np.testing.assert_array_equal(radius, r)
    with pytest.raises(ValueError, match='disjoint'):
        corridor(y, seen, [7], 'constant')


def test_holes_are_coherent_and_do_not_bridge_a_real_missing_run():
    y = np.column_stack((np.arange(35, dtype=float), np.zeros(35)))
    valid = np.ones(len(y), bool)
    valid[17:20] = False
    rows = pseudo_gaps(y, valid, key='agent')
    np.testing.assert_array_equal(rows['internal', 12, 'interpolation'], [1,12,12,12,1,1,0])
    assert ('terminal', 12, 'constant') not in rows
    np.testing.assert_array_equal(rows['terminal', 8, 'ols4'], [1,8,8,8,1,1,0])


def test_source_equal_summary_not_pooled():
    rows = np.array([[1,4,4,4,1,1,0], [3,12,0,4,0,1,12]])
    out = summary(rows, np.array([[0,0], [1,1]]))
    assert out['point_1x']['point'] == .5
    assert out['point_1x']['interval95'] == pytest.approx([.025,.975])
    assert out['gaps'] == 4
