import json

import numpy as np
import pytest

from src.evaluation.m3w_external_source_audit import (
    Track, audit_source, parse_citr, parse_gc, parse_hermes, parse_vru,
    parse_wildtrack_frame, track_stats,
)


def make_track(frames):
    frames = np.asarray(frames)
    return Track('recording', '1', 'pedestrian', frames, np.column_stack((frames, frames * 2)))


def test_exact_windows_never_interpolate_nondivisible_raw_horizon():
    stats = track_stats(make_track(np.arange(30) * 20), 20)
    assert stats['history8_exact_raw_horizon_windows'] == {'10': 0, '25': 0, '50': 0, '100': 18}
    assert stats['history_and_12step_future_windows']['8'] == 11


def test_gaps_break_history_and_future_support():
    stats = track_stats(make_track(np.r_[np.arange(15), np.arange(16, 31)]), 1)
    assert stats['discontinuous_edges'] == 1
    assert stats['history_and_12step_future_windows']['8'] == 0
    assert stats['history_only_windows']['8'] == 16


def test_identical_point_duplicate_deduped_but_conflict_rejected():
    track = make_track([0, 1, 1, 2])
    assert track_stats(track, 1)['duplicate_points'] == 1
    track.xy[2, 0] = 100
    with pytest.raises(ValueError, match='Conflicting'):
        track_stats(track, 1)


def test_bad_identity_and_nonfinite_positions_rejected():
    with pytest.raises(ValueError, match='noninteger'):
        track_stats(make_track([0, .5]), 1)
    track = make_track([0, 1])
    track.xy = track.xy.astype(float)
    track.xy[0, 0] = np.nan
    with pytest.raises(ValueError, match='positions'):
        track_stats(track, 1)


def test_gc_preserves_raw_axes_and_frames():
    track = parse_gc('525\n122\n0\n541\n141\n20\n', '0001')
    assert track.frame.tolist() == [0, 20]
    assert track.xy.tolist() == [[525, 122], [541, 141]]
    with pytest.raises(ValueError, match='triplets'):
        parse_gc('1\n2\n', 'bad')


def test_hermes_native_units_and_agents():
    tracks = parse_hermes('1 0 100 200 175\n2 0 400 500 180\n1 1 101 202 175', 'trial')
    assert len(tracks) == 2
    assert tracks[0].xy.tolist() == [[100, 200], [101, 202]]


def test_citr_vehicle_and_pedestrian_id_namespaces_are_separate():
    ped = parse_citr('frame,id,x,y,type\n0,1,2,3,ped', 'clip')[0]
    vehicle = parse_citr('frame,id,x_c,y_c,type\n0,1,9,10,veh', 'clip')[0]
    assert ped.agent != vehicle.agent
    assert vehicle.xy.tolist() == [[9, 10]]
    with pytest.raises(ValueError, match='filtered'):
        parse_citr('frame_id,agent_id,x_est,y_est\n0,1,9,10', 'clip')


def test_wildtrack_grid_uses_integer_row_not_fractional_division():
    rows = parse_wildtrack_frame(json.dumps([{'personID': 1, 'positionID': 481}]), 5)
    assert rows == [(1, 5, 1, 1)]
    with pytest.raises(ValueError, match='outside'):
        parse_wildtrack_frame('[{"personID":1,"positionID":691200}]', 5)


def test_vru_does_not_promote_measurement_index_to_raw_video_frame():
    rows = '\n'.join(f'{i},{i * .02},{i},{i * 2}' for i in range(30))
    track = parse_vru(',timestamp,x,y\n' + rows, 'object', 'pedestrian')
    stats = track_stats(track, 1, raw_frame_identity=False)
    assert stats['history8_exact_raw_horizon_windows'] is None
    assert stats['history_and_12step_future_windows']['8'] == 11
    assert stats['native_timestamp_start'] == 0
    track.timestamps[15:] += .02
    assert track_stats(track, 1, False)['discontinuous_edges'] == 1


def test_vru_bad_clock_rejected():
    track = parse_vru(',timestamp,x,y\n0,0,1,2\n1,0,2,3', 'object', 'pedestrian')
    with pytest.raises(ValueError, match='Nonincreasing'):
        track_stats(track, 1, False)


def test_real_audit_path_excludes_calibration_and_does_not_write_sources(tmp_path):
    directory = tmp_path / 'GC/Annotation'
    directory.mkdir(parents=True)
    (directory / '1.txt').write_text('1\n2\n0\n2\n3\n20\n')
    (directory.parent / 'H-world.txt').write_text('not a track')
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = audit_source(tmp_path, 'GC')
    assert result['raw_files'] == result['tracks_parsed'] == 1
    assert not result['official_split_assigned']
    assert not result['training_or_model_eval_run']
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_missing_source_is_not_success(tmp_path):
    result = audit_source(tmp_path, 'CITR')
    assert result['status'] == 'partial_or_missing_source_review_required'
    assert result['raw_files'] == 0


def test_hidden_version_control_metadata_is_not_trajectory(tmp_path):
    directory = tmp_path / 'VRU/pedestrians/.svn'
    directory.mkdir(parents=True)
    (directory / 'format.csv').write_text('not data')
    (directory.parent / 'track.csv').write_text(',timestamp,x,y\n0,0,1,2\n1,.02,2,3')
    result = audit_source(tmp_path, 'VRU')
    assert result['tracks_parsed'] == result['raw_files'] == 1
    assert result['excluded_hidden_metadata_files'] == ['pedestrians/.svn/format.csv']
    assert result['parse_or_identity_failures'] == 0


def test_invalid_stride_rejected():
    with pytest.raises(ValueError, match='stride'):
        track_stats(make_track([0, 1]), 0)


def test_window_counts_match_explicit_timestamp_support():
    rng = np.random.default_rng(917)
    for step in (1, 5, 20):
        for _ in range(12):
            frames = np.arange(130) * step
            frames = frames[rng.random(len(frames)) > .06]
            stats = track_stats(make_track(frames), step)
            support = set(frames.tolist())
            for history in (8, 16, 32, 64):
                count = sum(all(t in support for t in range(int(f - (history - 1) * step),
                            int(f + 12 * step + 1), step)) for f in frames)
                assert stats['history_and_12step_future_windows'][str(history)] == count
            for horizon in (10, 25, 50, 100):
                count = 0 if horizon % step else sum(all(t in support for t in
                    range(int(f - 7 * step), int(f + horizon + 1), step)) for f in frames)
                assert stats['history8_exact_raw_horizon_windows'][str(horizon)] == count
