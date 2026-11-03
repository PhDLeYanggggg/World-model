import numpy as np
import pytest

from src.evaluation.m3w_moving_relational import past_edges, score


def test_past_graph_translation_order_and_no_future_argument():
    xy = np.array([[0., 0.], [1, 0], [-1, 0], [0, 2]])
    ids = np.array([4, 3, 2, 1])
    edges = past_edges(xy, ids, neighbors=1)
    np.testing.assert_array_equal(edges, past_edges(xy+[1000, -1000], ids, neighbors=1))
    order = np.array([3, 0, 2, 1])
    shuffled = past_edges(xy[order], ids[order], neighbors=1)
    assert {tuple(sorted(v)) for v in edges} == {tuple(sorted(order[v])) for v in shuffled}
    with pytest.raises(TypeError):
        past_edges(xy, ids, future_mask=np.ones((4, 12), bool))


def test_perfect_pair_error_can_hide_arbitrarily_bad_absolute_error():
    y = np.zeros((2, 12, 2))
    y[1, :, 0] = np.arange(12)
    p = y+10000
    result = score(np.array([[0, 1]]), dict(cv=p, floor=p, pct=p, eqmotion=p), y, np.ones((2, 12), bool))
    assert result['metrics']['eqmotion']['available_pair_ADE_pixel'] == 0
    assert result['paired_known_pp'] is None
    assert result['common_translation_blind']
    assert np.linalg.norm(p-y, axis=-1).mean() > 10000


def test_unknown_pair_bounds_and_empty_graph_are_not_imputed_successes():
    y = np.zeros((3, 12, 2))
    valid = np.ones((3, 12), bool)
    valid[2] = False
    p = y.copy()
    p[0, :, 0] = 2
    e = y.copy()
    e[0, :, 0] = 1
    e[2, :, 0] = 5
    predictors = dict(cv=p, floor=p, pct=p, eqmotion=e)
    edges = np.array([[0, 1], [1, 2]])
    result = score(edges, predictors, y, valid)
    assert result['known_pairs'] == 1
    assert result['wholly_unknown_pairs'] == 1
    assert result['paired_known_pp'] > 0
    assert result['paired_completion_lower_pp'] < 0
    y[2] = -999999
    assert result == score(edges, predictors, y, valid)
    empty = score(np.empty((0, 2), int), predictors, y, valid)
    assert empty['edges'] == 0
    assert empty['paired_known_pp'] is None
    assert empty['metrics']['eqmotion']['available_pair_ADE_pixel'] is None
