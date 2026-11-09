import numpy as np
import pytest

from scripts.audit_m3w_ht21_detection_support import (
    box_member, detections, groups, labels_for, prefix, starts, support,
)


def box(identity, x):
    return [identity, x, 0., x+6., 6.]


def test_query_schedule_does_not_overlap_or_request_outside_clip():
    assert list(starts(19, 1)) == []
    assert list(starts(40, 1)) == [1, 21]
    assert list(starts(190, 10)) == []
    assert list(starts(191, 10)) == [1]
    assert list(starts(391, 10)) == [1, 201]


@pytest.mark.parametrize('bad', [b'0,-1,0,0,6,6,99,1,1,1', b'1,5,0,0,6,6,99,1,1,1',
                                 b'1,-1,0,0,0,6,99,1,1,1', b'1,-1,0,0,6,6,101,1,1,1'])
def test_detection_schema_is_fail_closed(bad):
    with pytest.raises(ValueError):
        detections(bad, 40)


def test_prefix_is_invariant_to_future_detection_change():
    rows = np.asarray([[f, -1, f, 0, 6, 6, 90, 1, 1, 1] for f in range(1, 21)], dtype=float)
    a = prefix(groups(rows), 1, 1)
    rows[8:, 2] += 100
    b = prefix(groups(rows), 1, 1)
    assert np.array_equal(a, b)
    assert len({a[j][0, 0] for j in range(8)}) == 1


def test_full_roster_keeps_partial_and_unmatched_tracks_in_denominator():
    online = [np.asarray([box(1, 0), box(2, 15), box(3, 30)]) for _ in range(8)]
    labels = [np.asarray([box(10, 0), box(20, 15), box(40, 55)]) for _ in range(20)]
    labels[-1] = labels[-1][[0, 2]]
    a = support(online, labels)
    assert (a['online'], a['complete'], a['partial'], a['unmatched']) == (3, 1, 1, 1)
    assert a['reference_complete'] == 2
    assert a['all_complete_multi_online'] == 0
    b = support([f[:2] for f in online], labels[:-1]+[labels[0]])
    assert b['all_complete_multi_online'] == 1


def test_association_must_preserve_identity_at_all_eight_observations():
    online = [np.asarray([box(1, 0), box(2, 15)]) for _ in range(8)]
    labels = [np.asarray([box(10, 0), box(20, 15)]) for _ in range(20)]
    labels[3] = np.asarray([box(20, 0), box(10, 15)])
    result = support(online, labels)
    assert result['online'] == result['unmatched'] == 2
    assert result['stable'] == 0


def test_archive_reader_refuses_publisher_test():
    with pytest.raises(ValueError):
        box_member(None, 'HT21-05', 'det')


def test_gt_ingestion_to_support_uses_identity_and_xyxy_not_frame_id():
    gt = ''.join(f'{f},15,{f},0,6,6,1,1,1\n{f},25,{f},0,6,6,1,3,1\n' for f in range(1, 21))
    by_frame = labels_for(gt.encode(), {'frames': 20})
    assert by_frame[1].tolist() == [box(15, 1)]
    online = [np.asarray([box(0, f)]) for f in range(1, 9)]
    result = support(online, [by_frame[f] for f in range(1, 21)])
    assert result['online'] == result['stable'] == result['complete'] == result['reference_complete'] == 1
