from contextlib import ExitStack

import numpy as np
import pytest

from scripts.audit_m3w_visdrone_camera_pilot import FRAMES, PastImages, label_frames, measure, select_clips
from scripts.audit_m3w_visdrone_label_scope import scoped_labels


def test_pilot_uses_first_eligible_clip_per_distinct_filename_group():
    roster = {f'uav{i:07}_00000_v': FRAMES for i in range(1, 6)}
    roster['uav0000001_00000_v'] = FRAMES[:-1]
    roster['uav0000001_00100_v'] = FRAMES
    roster['uav0000001_00200_v'] = FRAMES
    assert select_clips(roster) == ['uav0000001_00100_v', 'uav0000002_00000_v',
                                   'uav0000003_00000_v', 'uav0000004_00000_v']


def test_future_or_unregistered_images_are_rejected_before_access():
    roster = {f'uav{i:07}_00000_v': FRAMES for i in range(1, 6)}
    with ExitStack() as stack:
        reader = PastImages({'frames': roster}, stack, verify=True)
        with pytest.raises(ValueError, match='registered past'):
            reader.read('uav0000001_00000_v', FRAMES[8])
        with pytest.raises(ValueError, match='registered past'):
            reader.read('uav0000005_00000_v', FRAMES[0])
        assert reader.archive is None


def test_box_coverage_and_identity_support_are_distinct():
    raw = '\n'.join(f'{f},7,10,20,4,6,1,1,0,0\n{f},8,21,20,4,6,1,2,0,0' for f in FRAMES).encode()
    labels = label_frames(raw, FRAMES)
    assert np.array_equal(labels[0], [[7, 10, 20, 14, 26]])
    bundle = {'clip': 'synthetic', 'past': [{'online': [[i, 10, 20, 14, 26], [10, 21, 20, 25, 26]],
                                              'detections': [[0, 10, 20, 14, 26], [1, 21, 20, 25, 26]]}
                                             for i in range(8)]}
    report = measure(bundle, labels)
    assert report['reference_complete_detected_every_past_frame'] == report['reference_complete'] == 1
    assert report['complete'] == report['all_complete_multi_online'] == 0
    assert report['online'] == report['unmatched'] == 1


def test_label_scope_keeps_category_identity_and_excludes_ignored_rows():
    rows = np.array([[1, 7, 10, 20, 4, 6, 1, 1, 0, 0],
                     [1, 7, 21, 20, 4, 6, 1, 2, 0, 0],
                     [1, 9, 32, 20, 4, 6, 0, 2, 0, 0]])
    assert scoped_labels(rows, (1, 2))[0][:, 0].tolist() == [14, 15]
    assert scoped_labels(rows, (1,))[0][:, 0].tolist() == [14]
