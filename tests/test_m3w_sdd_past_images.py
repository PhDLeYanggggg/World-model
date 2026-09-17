import json

import numpy as np
import pytest

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_masked_history_images import masked_center_patch
from src.world_model.m3w_sdd_past_images import (
    ARRAYS, SDDPastImageStore, border_padding_suspect, supported_center_patch,
)


def test_black_border_is_suspect_but_isolated_dark_person_is_not():
    image = np.full((12, 12, 3), 100, np.uint8)
    image[:2] = 0
    image[5:7, 5:7] = 0
    suspect = border_padding_suspect(image)
    assert suspect[:2].all() and not suspect[2:].any()
    assert not suspect[5:7, 5:7].any()


def test_connectivity_and_threshold_fixed_not_learned_from_other_frames():
    image = np.full((4, 4, 3), 255, np.uint8)
    image[0, 0] = 8
    image[1, 1] = 0
    image[0, 3] = 9
    mask = border_padding_suspect(image, 8)
    assert mask.sum() == 1 and mask[0, 0]
    assert not border_padding_suspect(image, 7).any()


def test_supported_crop_keeps_both_observation_and_border_hypothesis():
    image = np.full((6, 6, 3), 90, np.uint8)
    image[:2] = 0
    patch = supported_center_patch(image, [1, 1], border_padding_suspect(image), 6, 2)
    old_rgb, old_count = masked_center_patch(image, [1, 1], 6, 2)
    np.testing.assert_array_equal(patch['rgb_observed'], old_rgb)
    np.testing.assert_array_equal(patch['geometric_count'], old_count)
    np.testing.assert_array_equal(patch['geometric_count'], [[1, 3], [3, 9]])
    np.testing.assert_array_equal(patch['retained_count'], [[0, 0], [2, 6]])
    assert patch['rgb_retained'][:, 1].min() == 90
    assert patch['rgb_retained'][:, 0].max() == 0


def test_interior_black_observation_has_positive_support():
    image = np.full((12, 12, 3), 255, np.uint8)
    image[3:9, 3:9] = 0
    a = supported_center_patch(image, [6, 6], border_padding_suspect(image), 6, 2)
    b = supported_center_patch(image, [-50, -50], border_padding_suspect(image), 6, 2)
    assert a['retained_count'].sum() == 36 and b['retained_count'].sum() == 0
    np.testing.assert_array_equal(a['rgb_retained'], b['rgb_retained'])


def test_all_black_frame_is_ambiguous_not_certified_missing():
    a = supported_center_patch(np.zeros((6, 6, 3), np.uint8), [3, 3],
                               border_padding_suspect(np.zeros((6, 6, 3), np.uint8)), 6, 2)
    assert a['geometric_count'].sum() == 36 and a['retained_count'].sum() == 0


@pytest.mark.parametrize('center,crop,output', [([np.nan, 0], 6, 2), ([0, 0], 6, 4),
                                               ([0, 0], True, 1), ([0, 0], 32, 1)])
def test_invalid_patch_inputs_rejected(center, crop, output):
    with pytest.raises(ValueError):
        supported_center_patch(np.zeros((6, 6, 3), np.uint8), center,
                               np.zeros((6, 6), bool), crop, output)


def make_store(path):
    n = 10
    keys = np.column_stack((np.arange(n), np.ones(n))).astype(np.int64)
    keys[-1] = [9, 2]
    flags = np.zeros((n, 3), np.uint8)
    flags[2, 0] = 1
    flags[3, 1] = 1
    flags[:, 2] = 1
    arrays = {'row_keys': keys, 'source_flags': flags,
        'annotation_boxes': np.tile([0., 0., 2., 2.], (n, 1)),
        'image_boxes': np.tile([0., 0., 1., 1.], (n, 1)),
        'rgb_observed': np.full((n, 3, 2, 2), 99, np.uint8),
        'rgb_retained': np.full((n, 3, 2, 2), 99, np.uint8),
        'geometric_count': np.full((n, 2, 2), 9, np.uint8),
        'retained_count': np.full((n, 2, 2), 9, np.uint8)}
    for name in ('rgb_observed', 'rgb_retained', 'geometric_count', 'retained_count'):
        arrays[name][2] = 0
    for name in ARRAYS:
        np.save(path/(name+'.npy'), arrays[name], allow_pickle=False)
    m = dict(data_role='diagnostic_only', observation_mode='offline_annotated', training_admitted=False,
             output_size=2, crop_size=6, decoded_prefix_frames=10,
             array_sha256={n: file_digest(path/(n+'.npy')) for n in ARRAYS})
    (path/'metadata.json').write_text(json.dumps(m))


def test_short_history_padded_not_dropped_or_recentered(tmp_path):
    make_store(tmp_path)
    s = SDDPastImageStore(tmp_path, observation_mode='offline_annotated')
    a = s.inputs(query_frame=3, agent_id=1, length=8)
    np.testing.assert_array_equal(a['source_frames'], np.arange(-4, 4))
    assert a['annotation_present_mask'].sum() == 4
    assert a['state_mask'].sum() == 3
    assert a['agent_unoccluded_flag'].sum() == 2
    assert a['geometric_count'][-1].sum() > 0  # Occluded context remains observed.
    assert a['geometric_count'][-2].sum() == 0  # Lost track provides no ghost crop.


def test_future_arrays_do_not_change_current_inputs_or_agent_population(tmp_path):
    make_store(tmp_path)
    s = SDDPastImageStore(tmp_path, observation_mode='offline_annotated')
    a = s.inputs(query_frame=7, agent_id=1)
    for name in ARRAYS:
        arr = np.array(s.arrays[name])
        arr[8:] = 0
        if name == 'row_keys':
            arr[8:] = [[100, 900], [101, 901]]
        s.arrays[name] = arr
    b = s.inputs(query_frame=7, agent_id=1)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_array_equal(s.agents_at(7), [1])
    assert not any('future' in name or 'target' in name for name in a)


@pytest.mark.parametrize('query,agent,length,step', [(10, 1, 8, 1), (-1, 1, 8, 1),
    (7, 2, 8, 1), (7, 1, 9, 1), (7, 1, 8, 0), (True, 1, 8, 1)])
def test_invalid_history_request_rejected(tmp_path, query, agent, length, step):
    make_store(tmp_path)
    s = SDDPastImageStore(tmp_path, observation_mode='offline_annotated')
    with pytest.raises(ValueError):
        s.inputs(query_frame=query, agent_id=agent, length=length, step=step)


def test_no_silent_training_admission_or_sensor_claim(tmp_path):
    make_store(tmp_path)
    with pytest.raises(ValueError, match='diagnostic'):
        SDDPastImageStore(tmp_path, observation_mode='offline_annotated', data_role='supervised_training')
    with pytest.raises(ValueError, match='diagnostic'):
        SDDPastImageStore(tmp_path, observation_mode='sensor_as_of')


def test_changed_cache_rejected(tmp_path):
    make_store(tmp_path)
    np.save(tmp_path/'source_flags.npy', np.ones((10, 3), np.uint8))
    with pytest.raises(ValueError, match='Changed image input'):
        SDDPastImageStore(tmp_path, observation_mode='offline_annotated')
