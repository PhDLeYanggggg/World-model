import json

import numpy as np
import pytest

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_masked_history_images import (
    ARRAYS, MaskedHistoryImageStore, masked_center_patch, validate_history_rows,
)


def test_full_support_block_mean_matches_direct_mean():
    image = np.arange(12 * 12 * 3, dtype=np.uint16).reshape(12, 12, 3).astype(np.uint8)
    rgb, mask = masked_center_patch(image, [6, 6], 6, 2)
    expected = np.rint(image[3:9, 3:9].reshape(2, 3, 2, 3, 3).mean((1, 3))).astype(np.uint8)
    np.testing.assert_array_equal(rgb.transpose(1, 2, 0), expected)
    np.testing.assert_array_equal(mask, np.full((2, 2), 9))


def test_partial_support_does_not_darkly_average_padding_or_recenter():
    image = np.full((12, 12, 3), [30, 60, 90], np.uint8)
    rgb, mask = masked_center_patch(image, [1, 1], 6, 2)
    np.testing.assert_array_equal(mask, [[1, 3], [3, 9]])
    np.testing.assert_array_equal(rgb, np.broadcast_to(np.array([30, 60, 90])[:, None, None], (3, 2, 2)))
    assert mask.sum() == 16


def test_black_observation_distinct_from_unobserved():
    image = np.zeros((12, 12, 3), np.uint8)
    black, black_mask = masked_center_patch(image, [6, 6], 6, 2)
    missing, missing_mask = masked_center_patch(image, [-100, -100], 6, 2)
    np.testing.assert_array_equal(black, missing)
    assert black_mask.sum() == 36 and missing_mask.sum() == 0


def test_nondivisible_and_nonfinite_patch_inputs_rejected():
    with pytest.raises(ValueError):
        masked_center_patch(np.zeros((6, 6, 3), np.uint8), [0, 0], 6, 4)
    with pytest.raises(ValueError):
        masked_center_patch(np.zeros((6, 6, 3), np.uint8), [np.nan, 0], 6, 2)


def make_store(path):
    n = 9
    arrays = {'rgb': np.full((n, 3, 2, 2), 100, np.uint8),
        'coverage': np.full((n, 2, 2), 9, np.uint8), 'native_xy': np.column_stack([np.arange(n), np.zeros(n)]),
        'image_xy': np.zeros((n, 2)), 'row_keys': np.column_stack([1 + np.arange(n)*10, np.ones(n)]),
        'source_frame': np.arange(n)*10, 'latest_control_frame': np.arange(n)*10,
        'history_rows': np.arange(8).reshape(1, 8), 'frame_decode_mask': np.ones(n, bool)}
    for name in ARRAYS:
        np.save(path / (name + '.npy'), arrays[name], allow_pickle=False)
    metadata = {'data_role': 'diagnostic_only', 'source_role': 'fit', 'formal_training_admitted': False,
        'crop_size': 6, 'output_size': 2, 'array_sha256': {n: file_digest(path / (n + '.npy')) for n in ARRAYS}}
    (path / 'metadata.json').write_text(json.dumps(metadata))


def test_reader_uses_only_selected_history_and_excludes_labels(tmp_path):
    make_store(tmp_path)
    store = MaskedHistoryImageStore(tmp_path, observation_mode='offline_annotation_diagnostic')
    a = store.inputs(0)
    # Unreferenced future row in resident arrays must not affect this query.
    altered = np.array(store.arrays['rgb'])
    altered[8] = 255
    store.arrays['rgb'] = altered
    altered_xy = np.array(store.arrays['native_xy'])
    altered_xy[8] = 100000
    store.arrays['native_xy'] = altered_xy
    store.arrays['future_endpoint'] = np.full((9, 2), -1e8)
    b = store.inputs(0)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert 'latest_control_frame' not in a and 'future_endpoint' not in a
    assert a['rgb'].shape == (8, 3, 2, 2)
    assert a['pixel_coverage'].shape == (8, 1, 2, 2)


def test_unapproved_role_and_later_controls_are_not_silently_admitted(tmp_path):
    make_store(tmp_path)
    with pytest.raises(ValueError, match='admission'):
        MaskedHistoryImageStore(tmp_path, observation_mode='offline_annotation_diagnostic', data_role='supervised_training')
    store = MaskedHistoryImageStore(tmp_path, observation_mode='control_as_of_query_diagnostic')
    late = np.array(store.arrays['latest_control_frame'])
    late[7] = 90
    store.arrays['latest_control_frame'] = late
    with pytest.raises(ValueError, match='after the query'):
        store.inputs(0)
    assert len(store) == 1


def test_changed_cache_rejected(tmp_path):
    make_store(tmp_path)
    np.save(tmp_path / 'rgb.npy', np.zeros((9, 3, 2, 2), np.uint8))
    with pytest.raises(ValueError, match='Changed image cache'):
        MaskedHistoryImageStore(tmp_path, observation_mode='offline_annotation_diagnostic')


def test_mixed_agent_or_future_history_rejected():
    keys = np.column_stack([1 + np.arange(9)*10, np.ones(9)])
    frames = np.arange(9)*10
    rows = np.arange(8).reshape(1, 8)
    validate_history_rows(keys, frames, rows)
    keys[3, 1] = 2
    with pytest.raises(ValueError, match='one agent'):
        validate_history_rows(keys, frames, rows)
    keys[3, 1] = 1
    rows[0, 4] = 8
    with pytest.raises(ValueError, match='exact past'):
        validate_history_rows(keys, frames, rows)
