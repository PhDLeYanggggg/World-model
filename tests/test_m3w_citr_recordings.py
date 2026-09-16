import hashlib
import json

import numpy as np
import pytest

from src.data_unification.m3w_causal_recordings import PROTOCOL_RAW
from src.data_unification.m3w_citr_recordings import (
    CITRRecordingWindows, PHYSICAL_SCENE, blob_hash, build_citr,
    check_past_only, parse_raw_object, source_snapshot,
)
from src.evaluation.m3w_recording_lineage import sha256


def raw_text(kind='p', n=130):
    if kind == 'p':
        return 'frame,id,x,y,type\n' + ''.join(f'{i},1,{i*.1},2,ped\n' for i in range(n))
    return 'frame,id,x_c,y_c,x_1,y_1,x_2,y_2,type\n' + ''.join(
        f'{i},1,{i*.2},3,0,0,1,1,veh\n' for i in range(n))


def fixture_source(tmp_path, clips=2):
    source = tmp_path / 'source'
    for clip in range(clips):
        directory = source / f'data/trajectories/scenario/clip_{clip:02d}'
        directory.mkdir(parents=True)
        for kind in ('p', 'v'):
            (directory / f'{kind}1.csv').write_text(raw_text(kind))
    manifest = [(str(p.relative_to(source)), blob_hash(p.read_bytes()))
                for p in sorted((source / 'data/trajectories').rglob('*.csv'))]
    report = {'status': 'verified', 'upstream_commit': 'synthetic_fixture_not_author_data',
              'raw_local_object_files': len(manifest),
              'raw_path_and_git_blob_manifest_sha256': hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
              'missing_local_count': 0, 'extra_local_count': 0, 'content_mismatch_count': 0}
    return source, tmp_path / 'derived', report


def test_namespace_does_not_merge_pedestrian_and_vehicle():
    ped, pm = parse_raw_object(raw_text(), 'p1.csv')
    veh, vm = parse_raw_object(raw_text('v'), 'v1.csv')
    assert set(ped[:, 1]) == {2} and set(veh[:, 1]) == {3}
    assert pm['source_id'] == vm['source_id'] == 1
    assert pm['agent_type'] == 'pedestrian' and vm['agent_type'] == 'vehicle'
    np.testing.assert_allclose(veh[:, 2], np.arange(130)*.2)


@pytest.mark.parametrize('text,name', [
    (raw_text(), 'p2.csv'), (raw_text(), 'v1.csv'), (raw_text(), 'filtered.csv'),
    (raw_text().replace('frame,id,x,y,type', 'frame,id,x,y,type,vx'), 'p1.csv'),
    (raw_text().replace(',ped', ',veh'), 'p1.csv'),
    (raw_text().replace('1,1,0.1,2,ped', '0,1,0.1,2,ped'), 'p1.csv'),
    (raw_text().replace('1,1,0.1,2,ped', '1,1,nan,2,ped'), 'p1.csv'),
    ('frame,id,x,y,type\n', 'p1.csv'),
    (raw_text().replace('1,1,0.1,2,ped', '1.5,1,0.1,2,ped'), 'p1.csv'),
], ids=['id_mismatch', 'schema_mismatch', 'filtered_filename', 'velocity_column',
        'type_mismatch', 'duplicate_frame', 'nonfinite_position', 'empty', 'fractional_frame'])
def test_invalid_or_filtered_objects_fail_closed(text, name):
    with pytest.raises(ValueError):
        parse_raw_object(text, name)


def test_real_row_mapping_and_roles_are_preserved(tmp_path):
    source, output, upstream = fixture_source(tmp_path)
    before = {p: sha256(p) for p in source.rglob('*.csv')}
    report = build_citr(source, output, upstream)
    assert report['result_source'] == 'fresh_run'
    assert report['points'] == 520 and report['agents'] == 4
    assert report['agent_types'] == {'pedestrian': 2, 'vehicle': 2}
    assert report['physical_scene_groups'] == [PHYSICAL_SCENE]
    assert not report['training_run'] and not report['official_split_assigned']
    assert not report['data_use_approval'] and not report['independent_confirmation']
    for record in report['recordings']:
        reader = CITRRecordingWindows(output / record['id'])
        assert reader.metadata['role'] == 'diagnostic_only'
        assert not reader.metadata['coordinate_conversion_applied']
        assert not reader.metadata['schema']['central_velocity_used']
        for file_id, entry in enumerate(reader.metadata['files']):
            path = source / entry['path']
            expected, _ = parse_raw_object(path.read_text(), path.name)
            rows = np.flatnonzero(reader.source_rows[:, 0] == file_id)
            np.testing.assert_array_equal(reader.points[rows], expected)
            np.testing.assert_array_equal(reader.source_rows[rows, 1], np.arange(2, 132))
        raw = reader.index[reader.index['protocol'] == PROTOCOL_RAW]
        np.testing.assert_array_equal(reader.points[raw['future_end'], 0] - reader.points[raw['current_row'], 0], raw['horizon_raw'])
    assert before == {p: sha256(p) for p in source.rglob('*.csv')}


def test_future_corruption_does_not_change_typed_scene_inputs(tmp_path):
    source, output, upstream = fixture_source(tmp_path)
    report = build_citr(source, output, upstream)
    check = check_past_only(output, report)
    assert check['future_label_api_calls'] == 0
    assert check['agent_queries_checked'] == 4
    assert check['agent_types_checked'] == {'pedestrian': 2, 'vehicle': 2}
    assert all(r['status'] == 'inputs_unchanged' for r in check['recording_checks'])


def test_gaps_retained_not_interpolated(tmp_path):
    source, output, _ = fixture_source(tmp_path, clips=1)
    p = next(source.rglob('p1.csv'))
    p.write_text('\n'.join(line for line in p.read_text().splitlines() if not line.startswith('60,'))+'\n')
    manifest = [(str(p.relative_to(source)), blob_hash(p.read_bytes())) for p in sorted(source.rglob('*.csv'))]
    upstream = {'status': 'verified', 'upstream_commit': 'fixture', 'raw_local_object_files': 2,
                'raw_path_and_git_blob_manifest_sha256': hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
                'missing_local_count': 0, 'extra_local_count': 0, 'content_mismatch_count': 0}
    report = build_citr(source, output, upstream)
    assert report['points'] == 259
    reader = CITRRecordingWindows(output / report['recordings'][0]['id'])
    for row in reader.index:
        span = reader.points[row['history_start']:row['future_end']+1]
        assert np.all(np.diff(span[:, 0]) == 1)


@pytest.mark.parametrize('change', ['modify', 'add', 'remove'])
def test_source_drift_rejected(tmp_path, change):
    source, output, upstream = fixture_source(tmp_path)
    path = next(source.rglob('p1.csv'))
    if change == 'modify':
        path.write_text(path.read_text().replace('0.1', '0.15'))
    elif change == 'add':
        path.with_name('p2.csv').write_text(raw_text())
    else:
        path.unlink()
    with pytest.raises(ValueError, match='snapshot'):
        source_snapshot(source, upstream)
    assert not output.exists()


def test_resume_verifies_without_overwriting_completed_clips(tmp_path):
    source, output, upstream = fixture_source(tmp_path)
    initial = build_citr(source, output, upstream)
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}
    report = build_citr(source, output, upstream, resume=True)
    assert report['result_source'] == 'cached_verified' and report['reused_recordings'] == 2
    assert report['points'] == initial['points']
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        build_citr(source, output, upstream)


@pytest.mark.parametrize('filename', ['metadata.json', 'source_rows.npy', 'points.npy', 'completion.json'])
def test_cache_drift_refuses_resume(tmp_path, filename):
    source, output, upstream = fixture_source(tmp_path, clips=1)
    report = build_citr(source, output, upstream)
    path = output / report['recordings'][0]['id'] / filename
    if filename.endswith('.json'):
        value = json.loads(path.read_text())
        value['run_identity_sha256' if filename == 'completion.json' else 'role'] = 'tampered'
        path.write_text(json.dumps(value))
    else:
        value = np.load(path)
        value[0, 0] += 1
        np.save(path, value)
    with pytest.raises(ValueError):
        build_citr(source, output, upstream, resume=True)


def test_completed_clip_survives_interruption(tmp_path):
    source, output, upstream = fixture_source(tmp_path)
    def stop(*_):
        raise InterruptedError('injected interruption after one atomic clip commit')
    with pytest.raises(InterruptedError):
        build_citr(source, output, upstream, progress=stop)
    report = build_citr(source, output, upstream, resume=True)
    assert report['reused_recordings'] == 1 and report['points'] == 520


def test_owned_partial_restarts_after_incomplete_array_write(tmp_path, monkeypatch):
    import src.data_unification.m3w_citr_recordings as converter
    source, output, upstream = fixture_source(tmp_path, clips=1)
    original = converter.write_recording
    def interrupt(directory, *args, **kwargs):
        (directory / 'incomplete.txt').write_text('partial array write')
        raise InterruptedError()
    monkeypatch.setattr(converter, 'write_recording', interrupt)
    with pytest.raises(InterruptedError):
        build_citr(source, output, upstream)
    monkeypatch.setattr(converter, 'write_recording', original)
    report = build_citr(source, output, upstream, resume=True)
    assert report['points'] == 260 and report['reused_recordings'] == 0
    assert not list(output.rglob('incomplete.txt'))


def test_typed_scene_membership_does_not_require_future_survival(tmp_path):
    source, output, upstream = fixture_source(tmp_path, clips=1)
    report = build_citr(source, output, upstream)
    reader = CITRRecordingWindows(output / report['recordings'][0]['id'])
    # Query the final frame, where neither object has a future label.
    scene = reader.get_scene_inputs(129, 50)
    assert {a['agent_type'] for a in scene['agents']} == {'pedestrian', 'vehicle'}
    assert all(not label['future_label_mask'].any() for label in reader.get_scene_labels(scene))
    assert all(not any('future' in k or 'remaining' in k or 'track_length' in k for k in a['inputs'])
               for a in scene['agents'])


def test_unowned_partial_directory_is_not_deleted(tmp_path):
    source, output, upstream = fixture_source(tmp_path)
    def stop(*_):
        raise InterruptedError()
    with pytest.raises(InterruptedError):
        build_citr(source, output, upstream, progress=stop)
    partial = output / 'citr_scenario__clip_01.partial'
    partial.mkdir()
    (partial / 'valuable.txt').write_text('preserve')
    with pytest.raises(ValueError, match='Unowned partial'):
        build_citr(source, output, upstream, resume=True)
    assert (partial / 'valuable.txt').read_text() == 'preserve'


def test_overlap_refused_before_writing(tmp_path):
    source, _, upstream = fixture_source(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        build_citr(source, source / 'derived', upstream)


def test_source_symlink_refused(tmp_path):
    source, _, upstream = fixture_source(tmp_path)
    path = next(source.rglob('p1.csv'))
    path.with_name('p2.csv').symlink_to(path)
    with pytest.raises(ValueError, match='symlink'):
        source_snapshot(source, upstream)
