import numpy as np

from scripts import audit_m3w_aerialmpt_detection_ceiling as ceiling
from scripts import run_m3w_aerialmpt_camera as study


def test_dense_tile_grid_covers_both_edges_without_gaps():
    for n in (425, 358):
        boxes = study.dense_tiles(n, 4)
        assert boxes[0][0] == 0 and sum(boxes[-1]) == n
        assert all(start < left+width for (left, width), (start, _) in zip(boxes, boxes[1:]))


def test_all_joint_online_outcomes_must_be_complete():
    frames = [dict(frame=t, pixel_sha256=str(t), boxes=[
        [t, 0, t+4, 4, .9, 1], [t, 15, t+4, 19, .9, 1]
    ]) for t in range(8)]
    window = study.confidence.tracks(frames, .3)
    labels = {t: {50: np.array([t+2., 2.]), 51: np.array([t+2., 17.])} for t in range(20)}
    full = study.assess(window, labels, 4.)
    assert full['online'] == full['stable'] == full['complete'] == 2
    assert full['fully_labeled_joint'] and full['complete_ade_px'] == [0., 0.]
    del labels[19][51]
    partial = study.assess(window, labels, 4.)
    assert partial['complete'] == 1 and not partial['fully_labeled_joint']
    assert partial['wholly_unknown'] == 0


def test_association_can_distinguish_id_switch_from_detection_miss():
    row = np.array([4, 0, 0, 4, 4, .9, 1])
    frames = [dict(frame=t, rows=[row.tolist()]) for t in range(8)]
    frames[-1]['rows'][0][0] = 5
    result = study.assess(dict(frames=frames, forecasts=[]),
                          {t: {50: np.array([2., 2.])} for t in range(20)}, 4.)
    assert result['eight_points_ignoring_online_id'] == 1 and result['stable'] == 0


def test_relaxation_allows_two_reference_tracks_to_use_one_detection():
    labels = {t: {1: np.array([2., 2.]), 2: np.array([3., 2.])} for t in range(20)}
    frames = [[[0., 0., 4., 4., .9, 1.]]] * 8
    assert ceiling.possible_histories(frames, labels, 4., .3) == {
        'histories': 2, 'histories_with_complete_future': 2}
    frames[-1] = []
    assert ceiling.possible_histories(frames, labels, 4., .3)['histories'] == 0
