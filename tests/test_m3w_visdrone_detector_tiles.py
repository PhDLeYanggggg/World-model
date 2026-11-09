import numpy as np
import pytest

from scripts.audit_m3w_visdrone_camera_pilot import FRAMES, OBS
from scripts.audit_m3w_visdrone_detector_tiles import person_boxes, pool, rows_for


def test_pool_deduplicates_cross_scale_boxes_but_keeps_new_coverage():
    coarse = np.array([[0, 0, 8, 8, .75, 1], [10, 0, 18, 8, .70, 1]])
    fine = np.array([[0, 0, 8, 8, .80, 1], [30, 0, 36, 8, .50, 1], [50, 0, 58, 8, .99, 2]])
    pooled = pool(coarse, person_boxes(fine))
    np.testing.assert_array_equal(pooled, np.stack((fine[0], coarse[1], fine[1])))
    assert pool(np.empty((0, 6)), np.empty((0, 6))).shape == (0, 6)


def test_all_arms_use_same_frozen_motion_and_coarse_boxes_must_match():
    past = list(FRAMES[:OBS])
    original = [dict(frame=t, detections=[[0, 20*i, 0, 20*i+8, 8]]) for i, t in enumerate(past)]
    moves = [dict(frame=t, matrix=[[1., 0., 20.], [0., 1., 0.], [0., 0., 1.]]) for t in past[1:]]
    scores = [dict(coarse=[[20*i, 0, 20*i+8, 8, .8, 1]],
                   fine=[[20*i, 1, 20*i+8, 9, .9, 1]]) for i in range(OBS)]
    arms = rows_for('clip', original, scores, moves)
    for k in arms:
        assert [r['online'][0][0] for r in arms[k]['past']] == [0] * OBS
    assert arms['pooled']['past'][0]['detections'] == arms['fine']['past'][0]['detections']
    scores[0]['coarse'][0][0] += 1
    with pytest.raises(ValueError, match='frozen coarse boxes'):
        rows_for('clip', original, scores, moves)
