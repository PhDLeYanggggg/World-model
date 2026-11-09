import json

import numpy as np
import pytest

from scripts import audit_m3w_sdd_camera_class_support as class_support
from scripts import audit_m3w_sdd_camera_confidence_utility as utility
from scripts import audit_m3w_sdd_camera_motion as motion
from scripts import audit_m3w_sdd_camera_coverage as coverage
from scripts import audit_m3w_sdd_camera_full_roster as full
from scripts import audit_m3w_sdd_camera_raster as raster
from scripts import audit_m3w_sdd_camera_unknown as unknown
from scripts import export_m3w_sdd_camera_donors as donors
from scripts import run_m3w_sdd_camera_aerial as aerial
from scripts import run_m3w_sdd_camera_confidence as confidence
from scripts import run_m3w_sdd_camera_scale_probe as scale
from scripts import run_m3w_sdd_camera_support as study


def test_roster_only_needs_eight_past_frames():
    visibility = {20*k + j: {1, 2} for k in range(10) for j in range(8)}
    assert study.candidate_windows(visibility, 200) == [20, 60, 120, 160]
    assert study.candidate_windows({j: {1, 2} for j in range(8)}, 19) == []


def test_paired_rows_require_stable_online_identity_and_mask_future():
    a, b = np.array([0, 0, 6, 6]), np.array([20, 20, 30, 30])
    labels = {t: {11: (a, True), 12: (b, False)} for t in range(20) if t != 19}
    frames = [dict(frame=t, rows=[[1, *a, .7, 1], [2 if t != 4 else 3, *b, .8, 1]]) for t in range(8)]
    empty = np.zeros((12, 2)).tolist()
    count, ids, rows = study.paired_window(dict(start=0, frames=frames, forecasts=[[1, empty, empty]]), labels)
    assert count['frame_matches'] == 16 and count['stable'] == 1 and count['complete'] == 0
    assert count['non_generated_future_steps'] == 11 and ids == {11}
    assert rows[0][-2].tolist() == [True] * 11 + [False]
    counts, bins = coverage.diagnostic(dict(frames=frames), labels)
    assert counts['publisher_ids_lost_to_id_changes'] == 1 and counts['stable'] == 1
    assert bins['under_10']['matched_boxes'] == bins['10_to_under_20']['matched_boxes'] == 8


def test_split_requires_train_in_both_and_rejects_overlap(tmp_path):
    a, b = (dict(scene_id='a', video_id=i) for i in range(2))
    first, second = tmp_path/'first.json', tmp_path/'second.json'
    first.write_text(json.dumps(dict(train=[a, b], val=[], test=[])))
    second.write_text(json.dumps(dict(train=[a], val=[], test=[b])))
    assert study.train_intersection([first, second]) == [('a', 0)]
    second.write_text(json.dumps(dict(train=[a], val=[b], test=[b])))
    with pytest.raises(ValueError, match='Overlap'):
        study.train_intersection([first, second])


def test_motion_compares_same_future_mask_with_a_separate_interpolation_slice():
    h = np.arange(8)[None, :, None] * np.array([2., -1.])[None, None, :]
    expected = np.arange(8, 20)[None, None, :, None] * np.array([2., -1.])[None, None, None, :]
    np.testing.assert_allclose(motion.forecasts(h), np.broadcast_to(expected, (1, 4, 12, 2)))
    mask = np.zeros((1, 12), dtype=bool)
    mask[:, -1] = True
    err = np.full((1, 2, 4, 12), 10.)
    err[:, :, :, -1] = 2.
    out = motion.report(err, mask, np.zeros_like(mask), [('clip', 1)])
    assert out['complete_ade_px'] is None and out['non_generated_step_mean_error_px'] is None
    assert out['available_endpoint_fde_px']['camera']['ols4'] == 2.
    assert out['known_step_mean_error_px']['publisher']['cv'] == 2.


def test_unmatched_online_history_is_retained_without_a_label():
    box = [0., 0., 2., 3., .9, 1.]
    future = np.zeros((12, 2)).tolist()
    window = dict(start=20, frames=[dict(rows=[[5, *box], [7, *box]]) for _ in range(8)],
                  forecasts=[[5, future, future], [7, future, future]])
    rows = list(donors.rows(window, {(20, 5): 'target'}))
    assert [x[0][1] for x in rows] == [5, 7]
    assert [x[1] for x in rows] == ['target', None]
    assert len(rows[1][0][2]) == 8


def test_scale_probe_restores_global_coordinates_and_suppresses_duplicate_boxes():
    image = np.zeros((90, 90, 3), dtype=np.uint8)
    image[43:47, 43:47] = 255

    def detector(tile):
        y, x = np.nonzero(tile[..., 0])
        return np.array([[x.min(), y.min(), x.max()+1, y.max()+1, .8, 1.]])

    np.testing.assert_array_equal(scale.tiled(detector, image), [[43, 43, 47, 47, .8, 1]])


def test_aerial_decode_unletterboxes_people_and_rejects_nonperson():
    image = np.zeros((120, 200, 3), np.uint8)
    image[:, :, 2] = 255
    x, (dx, dy, sx, sy) = aerial.letterbox(image)
    assert x.shape == (1, 3, 768, 1344) and x[0, 0, 384, 672] == 1.
    raw = np.zeros((1, 14, 2))
    raw[0, :4] = np.array([(70*sx+dx)/1344, (50*sy+dy)/768, 20*sx/1344, 30*sy/768])[:, None]
    raw[0, 4, 0], raw[0, 6, 1] = .9, .95
    np.testing.assert_allclose(aerial.decode(raw, (dx, dy, sx, sy), image.shape), [[60, 35, 80, 65, .9, 1.]])


def test_raster_map_is_an_xy_unit_conversion_for_past_and_future():
    raw = {7: {10: (np.array([120, 60, 140, 80]), True)},
           8: {10: (np.array([124, 64, 144, 84]), False)}}
    d = raster.convert(raw, (.5, .75))
    assert raster.outside(raw, (90, 100)) == 2 and raster.outside(d, (90, 100)) == 0
    np.testing.assert_allclose(d[8][10][0], [62, 48, 72, 63])
    assert d[8][10][1] is False and raw[7][10][0][0] == 120


def test_full_roster_reattaches_only_matched_outcomes():
    x = dict(start=np.array([20, 20]), online_id=np.array([1, 2]),
             past_xyxy_conf_class=np.zeros((2, 8, 6)), cv_ols4=np.zeros((2, 2, 12, 2)))
    z = {k: v[1:] for k, v in x.items()}
    z.update(target_xy=np.ones((1, 12, 2)), target_mask=np.ones((1, 12), bool),
             non_generated_mask=np.zeros((1, 12), bool))
    out = full.attach(x, z)
    assert out['stable_match'].tolist() == [False, True]
    assert not out['target_mask'][0].any() and out['target_mask'][1].all()
    with pytest.raises(ValueError, match='subset'):
        full.attach(x, {**z, 'online_id': np.array([3])})


def test_unknown_rows_debit_gain_and_empty_reference_is_not_zero():
    f = np.zeros((2, 4, 12, 2))
    f[0, 0, :, 0] = 1.
    f[1, 0, :, 0] = 4.
    y = np.zeros((2, 12, 2))
    valid = np.broadcast_to([[True], [False]], (2, 12))
    weight = np.array([.5, .5])
    d = full.readout(f, y, valid, weight)['contrasts']['ols4_vs_cv']
    assert d == dict(known_pp=100., lower_pp=-300., missing_debit_px=24.)
    assert full.readout(f, y, ~np.ones_like(valid), weight)['contrasts']['ols4_vs_cv']['lower_pp'] is None


def test_confidence_bridge_preserves_strong_tracks_and_requires_anchors():
    frames = [
        dict(frame=t, pixel_sha256=str(t), boxes=[
            [t, 0, t+6, 6, .9, 1],
            [t, 20, t+6, 26, .9 if t in (0, 3, 7) else .2, 1],
            [t, 40, t+6, 46, .9 if t in (0, 7) else .2, 1],
        ]) for t in range(8)
    ]
    high, low = (confidence.tracks(frames, x) for x in (.30, .12))
    added = confidence.bridge(high, low)
    assert [len(w['forecasts']) for w in (high, low, added)] == [1, 3, 2]
    assert added['forecasts'][0] == high['forecasts'][0]
    assert added['forecasts'][1][1:] == low['forecasts'][1][1:]
    assert all(len(f['rows']) == 2 for f in added['frames'])
    values, matched = utility.roster(added, {})
    assert values[0].shape == (2, 4, 12, 2)
    assert values[3].tolist() == [.5, .5] and not values[2].any() and not matched
    empty = confidence.tracks([dict(f, boxes=[]) for f in frames], .30)
    assert utility.roster(empty, {})[0][0].shape == (0, 4, 12, 2)


def test_unknown_taxonomy_distinguishes_nonpedestrians_and_missing_outcomes():
    a, b, c = (np.array([x, 0, x+6, 6]) for x in (0, 20, 40))
    boxes = [a, b, c]
    frames = [dict(frame=t, rows=[[i, *box, .9, 1] for i, box in enumerate(boxes)]) for t in range(8)]
    window = dict(start=0, frames=frames, forecasts=[[i, [], []] for i in range(3)])
    labels = {t: {10: (a, True), **({11: (b, True)} if t < 8 else {})} for t in range(20)}
    count, classes = unknown.reasons(window, labels, {10: 'Pedestrian', 11: 'Biker'})
    assert count == dict(scoring_complete=1, stable_nonpedestrian=1, never_matched_visible_label=1)
    assert classes == dict(Biker=1)


def test_class_support_reserves_pedestrian_and_keeps_unknown_joint_track():
    boxes = np.array([[0, 0, 6, 6], [20, 0, 26, 6], [40, 0, 46, 6]])
    frames = [dict(frame=t, pixel_sha256=str(t), boxes=[[*b, .9, 1] for b in boxes]) for t in range(8)]
    window = confidence.tracks(frames, .30)
    labels = {t: {10: (boxes[0] + [1, 0, 1, 0], True), 11: (boxes[0], True),
                  12: (boxes[1], False)} for t in range(20)}
    classes = {10: 'Pedestrian', 11: 'Biker', 12: 'Skater'}
    values, group = class_support.score_window(window, labels, classes, 'test_clip')
    assert set(group) == {'Pedestrian', 'Skater'}
    assert values[2].all(1).tolist() == [True, True, False]
    assert class_support.window_support(values, group) == {
        'online_at_least_two': 1, 'complete_at_least_one': 1, 'complete_at_least_two': 1,
        'complete_all_online_at_least_two': 0, 'complete_at_least_two_classes': 1,
    }
