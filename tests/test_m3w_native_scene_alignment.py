import numpy as np
import pytest

from src.evaluation.m3w_native_scene_alignment import (
    restore, past_transforms, scene_index, resolve_target_neighbors, support_counts,
)


def fixture():
    h = np.stack((np.column_stack((np.arange(8)*3., np.arange(8)*4.)),
                  np.column_stack((np.ones(8)*9., np.ones(8)*8.))))
    boxes = np.concatenate((h-1, h+1), -1).reshape(-1, 4)
    crops = np.array([(f*12, a) for a in range(2) for f in range(8)])
    rows = np.arange(16).reshape(2, 8)
    keys = np.array([[84, 0], [84, 1]])
    rotation = np.array([[[.6, -.8], [.8, .6]], np.eye(2)])
    scale = np.array([60., .001])
    g = np.zeros((2, 476), np.float32)
    g[:, :16] = (np.einsum('nti,nij->ntj', h-h[:, -1, None], rotation)/scale[:, None, None]).reshape(2, 16)
    g[:, 16:24] = -np.arange(7, -1, -1)/12
    cv = (h[:, -1]-h[:, -2])[:, None]*np.arange(1, 13)[None, :, None]
    g[:, 332:356] = (np.einsum('nti,nij->ntj', cv, rotation)/scale[:, None, None]).reshape(2, 24)
    return keys, rows, crops, boxes, g, scale.astype(np.float32).astype(float)


def test_transform_reconstructs_rotation_and_stop():
    result = past_transforms(*fixture())
    assert result['max_history_error_stored_scale'] < 1e-5
    np.testing.assert_allclose(result['origin'], [[21, 28], [9, 8]])
    assert result['max_scale_relative_rounding'] < 1e-7


@pytest.mark.parametrize('fault', ['future_crop', 'wrong_agent', 'duplicate', 'bad_scale', 'future_cv'])
def test_rejects_provenance_and_rollout_errors(fault):
    keys, rows, crops, boxes, g, s = fixture()
    if fault == 'future_crop': crops[0, 0] = 96
    if fault == 'wrong_agent': crops[0, 1] = 13
    if fault == 'duplicate': keys[1] = keys[0]
    if fault == 'bad_scale': s[0] *= 2
    if fault == 'future_cv': g[0, 332] += 1
    with pytest.raises((ValueError, AssertionError)):
        past_transforms(keys, rows, crops, boxes, g, s)


def test_metric_invariance():
    result = past_transforms(*fixture())
    p = np.random.default_rng(7).normal(size=(2, 12, 2))
    q = p+1
    args = (result['origin'], result['rotation'], result['stored_metric_scale'])
    before = np.linalg.norm(p-q, axis=-1)*args[-1][:, None]
    after = np.linalg.norm(restore(p, *args)-restore(q, *args), axis=-1)
    np.testing.assert_allclose(before, after, atol=1e-12)


def test_recordings_and_times_never_mixed():
    idx = scene_index(np.array(['b', 'a', 'a', 'a']), np.array([12, 12, 24, 12]), np.array(['1', '1', '1', '2']))
    np.testing.assert_array_equal(idx['order'], [1, 3, 2, 0])
    np.testing.assert_array_equal(idx['offsets'], [0, 2, 3, 4])
    with pytest.raises(ValueError, match='Duplicate'):
        scene_index(np.array(['a', 'a']), np.array([12, 12]), np.array(['1', '1']))


def test_ambiguous_neighbors_not_silently_resolved():
    g = np.zeros((3, 476), np.float32)
    origin = np.array([[0., 0.], [1., 0.], [1., 0.]])
    g[0, 52:54] = [1., 0.]
    g[0, 237] = 1
    idx = scene_index(np.array(['a']*3), np.array([12]*3), np.array(['1', '2', '3']))
    result = resolve_target_neighbors(g, origin, np.tile(np.eye(2), (3, 1, 1)), np.ones(3), idx)
    assert result['target_row'][0, 0] == -2
    assert np.all(result['target_row'][1:] == -1)
    origin[2] = [2, 0]
    result = resolve_target_neighbors(g, origin, np.tile(np.eye(2), (3, 1, 1)), np.ones(3), idx)
    assert result['target_row'][0, 0] == 1


def test_unknown_outcomes_retained():
    mask = np.zeros((3, 12), bool); mask[0] = True; mask[1, 0] = True
    r = support_counts(np.ones(3, bool), mask, np.array([0, 0, 1]), np.array(['a', 'a', 'b']))
    assert (r['complete_selected'], r['partial_selected'], r['unknown_selected']) == (1, 1, 1)
    assert r['unique_selected_tracks'] == 2
    assert r['selected_scene_queries'] == 2
