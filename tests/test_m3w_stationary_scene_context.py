from copy import deepcopy

import numpy as np
import pytest

from src.evaluation.m3w_stationary_scene_context import (
    obstacle_frame, scene_context, encode_future_label, forecast_metrics, read_obstacles,
)
from tests.test_m3w_stationary_start_probe import inputs


def geometry():
    return {'lines': np.array([[[-4., -2.], [5., -2.]], [[5., -2.], [5., 7.]]]),
            'circles': np.array([[-3., 5., .3]])}


def test_scene_features_and_labels_rotate_scale_translate_consistently():
    x, center, geo = inputs(), np.array([1., 0.]), geometry()
    features, frame = scene_context(x, center, geo)
    future = center+np.column_stack([np.arange(1, 13)*.1, np.zeros(12)])
    label = encode_future_label(future, center, frame)
    rot = np.array([[0., -1.], [1., 0.]])
    shift, factor = np.array([5., -2.]), 17.
    changed = deepcopy(x)
    changed['neighbor_xy'] = changed['neighbor_xy'] @ rot*factor
    transformed = {'lines': geo['lines'] @ rot*factor+shift,
                   'circles': np.column_stack([geo['circles'][:, :2] @ rot*factor+shift, geo['circles'][:, 2]*factor])}
    f2, frame2 = scene_context(changed, center @ rot*factor+shift, transformed)
    y2 = encode_future_label(future @ rot*factor+shift, center @ rot*factor+shift, frame2)
    np.testing.assert_allclose(features, f2, atol=1e-10)
    np.testing.assert_allclose(label, y2, atol=1e-10)
    restored = label @ frame['basis'].T*frame['scale']
    np.testing.assert_allclose(restored, future-center, atol=1e-12)


def test_future_key_rejected_and_future_values_do_not_change_features():
    x = inputs()
    f, frame = scene_context(x, [1, 0], geometry())
    encode_future_label(np.ones((12, 2))*900, [1, 0], frame)
    np.testing.assert_array_equal(f, scene_context(x, [1, 0], geometry())[0])
    x['future_endpoint'] = [1, 0]
    with pytest.raises(ValueError, match='schema'):
        scene_context(x, [1, 0], geometry())


def test_ambiguous_or_zero_distance_frame_does_not_invent_direction():
    geo = {'lines': np.array([[[-3., -1.], [3., -1.]], [[-3., 1.], [3., 1.]]]),
           'circles': np.zeros((0, 3))}
    frame = obstacle_frame([0, 0], geo)
    assert not frame['defined']
    np.testing.assert_array_equal(frame['basis'], 0)
    assert not obstacle_frame([0, 1], geo)['defined']


def test_duplicate_closest_corner_is_not_an_ambiguous_direction():
    geo = {'lines': np.array([[[0., 0.], [3., 0.]], [[3., 0.], [3., 3.]]]),
           'circles': np.zeros((0, 3))}
    frame = obstacle_frame([4, -1], geo)
    assert frame['defined'] and frame['features'][-1] == 0
    np.testing.assert_allclose(frame['basis'][:, 0], [1/np.sqrt(2), -1/np.sqrt(2)])


def test_xml_namespace_and_bad_geometry(tmp_path):
    path = tmp_path/'map.xml'
    path.write_text('<Trial xmlns="test"><Line x1="0" y1="0" x2="1" y2="0"/><Circle x="2" y="3" radius="0.2"/></Trial>')
    g = read_obstacles(path)
    assert g['lines'].shape == (1, 2, 2) and g['circles'].shape == (1, 3)
    g['circles'][0, 2] = -1
    with pytest.raises(ValueError):
        obstacle_frame([0, 0], g)


def test_zero_floor_easy_harm_not_hidden_and_direction_exclusions_reported():
    target = np.zeros((2, 12, 2)); target[1, :, 0] = 1
    p = target.copy(); p[0, :, 0] = .1
    m = forecast_metrics(p, target, np.ones(2), .2, [('a',), ('b',)])
    assert m['easy_rows'] == 1 and m['easy_absolute_harm'] == pytest.approx(.1)
    assert m['easy_degradation_pct'] is None
    assert m['direction_scored_rows'] == m['true_nonzero_endpoint_rows'] == 1
    assert m['angular_error_degrees'] == pytest.approx(0)
    assert not m['group_balanced']['independence_established']


def test_summary_rejects_missing_frozen_models():
    from scripts.summarize_m3w_stationary_scene_probe import summarize_trials
    with pytest.raises(ValueError, match='exactly once'):
        summarize_trials([])
