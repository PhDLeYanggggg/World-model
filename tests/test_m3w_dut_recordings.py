import hashlib
import json

import numpy as np
import pytest

from scripts.fetch_m3w_dut_annotations import ALLOWLIST, TREE, select_tree, safe_path, verify_bytes, blob_hash
from src.data_unification.m3w_dut_recordings import build_dut, DUTRecordingWindows, parse_raw_table, check_past_only
from src.evaluation.m3w_recording_lineage import sha256


def raw_text(kind='ped', n=130):
    if kind == 'ped':
        return 'id,x,y,frame,label\n' + ''.join(f'{a},{i*.1+a},2,{i},ped\n' for i in range(n) for a in (0, 1))
    return 'id,x_c,y_c,x_fl,y_fl,x_fr,y_fr,x_rr,y_rr,x_rl,y_rl,frame,label\n' + ''.join(
        f'0,{i*.2},3,0,0,1,0,1,1,0,1,{i},veh\n' for i in range(n))


def source(tmp_path):
    root, out = tmp_path / 'source', tmp_path / 'cache'
    for clip in ('intersection_01', 'roundabout_01'):
        for kind in ('ped', 'veh'):
            path = root / f'data/trajectories/{clip}_traj_{kind}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw_text(kind))
    files = [{'path': str(p.relative_to(root)), 'sha': blob_hash(p.read_bytes()), 'sha256': sha256(p), 'size': p.stat().st_size}
             for p in sorted(root.rglob('*.csv'))]
    report = {'status': 'verified', 'upstream_commit': 'synthetic_fixture', 'files': files,
              'raw_local_csv_files': len(files), 'raw_path_and_git_blob_manifest_sha256': hashlib.sha256(json.dumps([(e['path'], e['sha']) for e in files]).encode()).hexdigest()}
    return root, out, report


def test_raw_interleaved_rows_and_typed_identity():
    points, agents = parse_raw_table(raw_text(), 'intersection_01_traj_ped.csv')
    assert set(points[:, 1]) == {0, 2}
    assert points.shape == (260, 4)
    assert set(agents) == {'0', '2'}
    vehicle, _ = parse_raw_table(raw_text('veh'), 'intersection_01_traj_veh.csv')
    assert set(vehicle[:, 1]) == {1}
    np.testing.assert_array_equal(vehicle[:, 2], np.arange(130)*.2)


@pytest.mark.parametrize('text,name', [
    (raw_text().replace('x,y', 'x_est,y_est'), 'intersection_01_traj_ped.csv'),
    (raw_text(), 'intersection_01_traj_veh.csv'),
    (raw_text(), 'intersection_01_traj_ped_filtered.csv'),
    (raw_text().replace(',ped', ',veh'), 'intersection_01_traj_ped.csv'),
    (raw_text().replace('0,0.0,2,0,ped', '0,nan,2,0,ped'), 'intersection_01_traj_ped.csv'),
    (raw_text().replace('0,0.0,2,0,ped', '0.5,0.0,2,0,ped'), 'intersection_01_traj_ped.csv'),
    (raw_text().replace('0,0.1,2,1,ped', '0,0.1,2,0,ped'), 'intersection_01_traj_ped.csv'),
    ('id,x,y,frame,label\n', 'intersection_01_traj_ped.csv'),
])
def test_bad_or_filtered_rows_refused(text, name):
    with pytest.raises(ValueError):
        parse_raw_table(text, name)


def test_convert_preserves_every_raw_row_and_no_formal_role(tmp_path):
    root, out, upstream = source(tmp_path)
    result = build_dut(root, out, upstream)
    assert result['points'] == 780 and result['agents'] == 6
    assert result['agent_types'] == {'pedestrian': 4, 'vehicle': 2}
    assert result['physical_scene_groups'] == ['dut_intersection', 'dut_shared_space']
    assert not result['official_split_assigned'] and not result['training_run']
    for r in result['recordings']:
        reader = DUTRecordingWindows(out / r['id'])
        assert reader.metadata['coordinate_claim'] == 'dataset_local_unverified'
        assert reader.metadata['role'] == 'diagnostic_only'
        for i, f in enumerate(reader.metadata['files']):
            expected, _ = parse_raw_table((root / f['path']).read_text(), f['path'].split('/')[-1])
            rows = np.flatnonzero(reader.source_rows[:, 0] == i)
            np.testing.assert_array_equal(reader.points[rows], expected[reader.source_rows[rows, 1]-2])
    check = check_past_only(out, result)
    assert check['future_label_api_calls'] == 0
    assert check['agent_queries_checked'] == 6
    assert check['checks_passed'] == 2


def test_resume_and_source_drift(tmp_path):
    root, out, upstream = source(tmp_path)
    build_dut(root, out, upstream)
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    resumed = build_dut(root, out, upstream, resume=True)
    assert resumed['result_source'] == 'cached_verified' and resumed['reused_recordings'] == 2
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    next(root.rglob('*.csv')).write_text('tampered')
    with pytest.raises(ValueError, match='source|snapshot'):
        build_dut(root, out, upstream, resume=True)


@pytest.mark.parametrize('filename', ['points.npy', 'source_rows.npy', 'metadata.json', 'completion.json'])
def test_derived_drift_refused(tmp_path, filename):
    root, out, upstream = source(tmp_path)
    result = build_dut(root, out, upstream)
    path = out / result['recordings'][0]['id'] / filename
    if filename.endswith('.json'):
        x = json.loads(path.read_text())
        x['run_identity_sha256' if filename == 'completion.json' else 'role'] = 'tampered'
        path.write_text(json.dumps(x))
    else:
        x = np.load(path)
        x.flat[0] += 1
        np.save(path, x)
    with pytest.raises(ValueError):
        build_dut(root, out, upstream, resume=True)


def test_tree_allowlist_rejects_truncation_symlinks_missing_and_large():
    tree = {'sha': TREE, 'truncated': False, 'tree': [{'path': p, 'sha': 'a'*40, 'size': 1, 'mode': '100644', 'type': 'blob'} for p in ALLOWLIST]}
    assert len(select_tree(tree)) == len(ALLOWLIST)
    for changed in ({**tree, 'truncated': True}, {**tree, 'sha': 'b'*40}, {**tree, 'tree': tree['tree'][:-1]}):
        with pytest.raises(ValueError):
            select_tree(changed)
    tree['tree'][0]['mode'] = '120000'
    with pytest.raises(ValueError):
        select_tree(tree)
    tree['tree'][0]['mode'] = '100644'
    tree['tree'][0]['size'] = 40_000_000
    with pytest.raises(ValueError):
        select_tree(tree)


def test_hash_and_destination_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        verify_bytes(b'abc', {'size': 3, 'sha': 'a'*40})
    with pytest.raises(ValueError):
        safe_path(tmp_path, '../outside.csv')
    (tmp_path / 'data').symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(ValueError):
        safe_path(tmp_path, 'data/trajectories/intersection_01_traj_ped.csv')


def test_interrupted_conversion_resumes_without_touching_completed_clip(tmp_path):
    root, out, upstream = source(tmp_path)
    def interrupt(*args):
        raise InterruptedError('after completed clip')
    with pytest.raises(InterruptedError):
        build_dut(root, out, upstream, progress=interrupt)
    first = out / 'dut_intersection_01' / 'points.npy'
    saved = sha256(first), first.stat().st_mtime_ns
    result = build_dut(root, out, upstream, resume=True)
    assert result['reused_recordings'] == 1
    assert saved == (sha256(first), first.stat().st_mtime_ns)


def test_owned_partial_rebuild_and_unowned_partial_preservation(tmp_path, monkeypatch):
    import src.data_unification.m3w_dut_recordings as module
    root, out, upstream = source(tmp_path)
    original = module.write_recording
    def interrupt(directory, *args):
        (directory / 'partial.txt').write_text('incomplete array')
        raise InterruptedError()
    monkeypatch.setattr(module, 'write_recording', interrupt)
    with pytest.raises(InterruptedError):
        build_dut(root, out, upstream)
    monkeypatch.setattr(module, 'write_recording', original)
    partial = out / 'dut_roundabout_01.partial'
    partial.mkdir()
    (partial / 'preserve.txt').write_text('not our run')
    with pytest.raises(ValueError, match='Unowned partial'):
        build_dut(root, out, upstream, resume=True)
    assert (partial / 'preserve.txt').read_text() == 'not our run'
    assert not (out / 'dut_intersection_01.partial').exists()
    assert (out / 'dut_intersection_01/completion.json').exists()


def test_future_survival_is_not_input_membership(tmp_path):
    root, out, upstream = source(tmp_path)
    result = build_dut(root, out, upstream)
    reader = DUTRecordingWindows(out / result['recordings'][0]['id'])
    inputs = reader.get_scene_inputs(129, 50)
    assert len(inputs['agents']) == 3
    assert all(not x['future_label_mask'].any() for x in reader.get_scene_labels(inputs))
    assert all(not any(t in key for t in ('future', 'remaining', 'track_length'))
               for a in inputs['agents'] for key in a['inputs'])


def test_native_gaps_kept_and_source_row_alignment_still_exact(tmp_path):
    root, out, upstream = source(tmp_path)
    path = root / upstream['files'][0]['path']
    path.write_text('\n'.join(line for line in path.read_text().splitlines() if not line.endswith(',60,ped'))+'\n')
    e = upstream['files'][0]
    e.update(sha=blob_hash(path.read_bytes()), sha256=sha256(path), size=path.stat().st_size)
    upstream['raw_path_and_git_blob_manifest_sha256'] = hashlib.sha256(json.dumps([(e['path'],e['sha']) for e in upstream['files']]).encode()).hexdigest()
    result = build_dut(root, out, upstream)
    assert result['points'] == 778
    reader = DUTRecordingWindows(out / result['recordings'][0]['id'])
    for r in reader.index:
        frames = reader.points[r['history_start']:r['future_end']+1, 0]
        assert len(set(np.diff(frames))) == 1


def test_fetch_resume_uses_identical_existing_files_and_never_overwrites(tmp_path, monkeypatch):
    import scripts.fetch_m3w_dut_annotations as fetcher
    path = 'data/trajectories/intersection_01_traj_ped.csv'
    data = raw_text().encode()
    entry = {'path': path, 'sha': blob_hash(data), 'size': len(data)}
    monkeypatch.setattr(fetcher, 'RAW_PATHS', {path})
    calls = []
    def download(url, bound):
        calls.append((url, bound))
        return data
    _, written, reused = fetcher.acquire(tmp_path, [entry], download)
    assert (written, reused, len(calls)) == (1, 0, 1)
    _, written, reused = fetcher.acquire(tmp_path, [entry], download)
    assert (written, reused, len(calls)) == (0, 1, 1)
    p = tmp_path / path
    p.write_text('changed')
    with pytest.raises(ValueError, match='pinned author blob'):
        fetcher.acquire(tmp_path, [entry], download)
    assert p.read_text() == 'changed'


def test_code_identity_change_refuses_completed_resume(tmp_path, monkeypatch):
    import src.data_unification.m3w_dut_recordings as module
    root, out, upstream = source(tmp_path)
    build_dut(root, out, upstream)
    original = module.sha256
    monkeypatch.setattr(module, 'sha256', lambda p: 'changed' if str(p).endswith('m3w_dut_recordings.py') else original(p))
    with pytest.raises(ValueError, match='identity changed'):
        build_dut(root, out, upstream, resume=True)


@pytest.mark.parametrize('frames', [np.arange(130), np.arange(0, 150, 10), np.r_[np.arange(70), np.arange(71,130)], np.r_[np.arange(60),np.arange(60,180,10)]])
def test_independent_uniform_run_recount_matches_explicit_grid(frames):
    from scripts.audit_m3w_dut_intake import count_windows
    counts, steps = count_windows(frames)
    brute = {str(h): 0 for h in (10,25,50,100)}
    obs = 0
    for i in range(7, len(frames)):
        for j in range(i+1, len(frames)):
            if len(set(np.diff(frames[i-7:j+1]))) != 1:
                continue
            raw = int(frames[j]-frames[i])
            if str(raw) in brute:
                brute[str(raw)] += 1
            obs += j-i == 12
    assert counts == brute and steps == obs


@pytest.mark.parametrize('duplicate_within_clip', [False, True])
def test_independent_audit_writes_serializable_evidence_and_detects_duplicates(tmp_path, monkeypatch, duplicate_within_clip):
    import scripts.audit_m3w_dut_intake as module
    root, out, upstream = source(tmp_path)
    if duplicate_within_clip:
        path = root / upstream['files'][0]['path']
        path.write_text('id,x,y,frame,label\n' + ''.join(f'{a},{i*.1},2,{i},ped\n' for i in range(130) for a in (0,1)))
        upstream['files'][0].update(sha=blob_hash(path.read_bytes()), sha256=sha256(path), size=path.stat().st_size)
        upstream['raw_path_and_git_blob_manifest_sha256'] = hashlib.sha256(json.dumps([(e['path'],e['sha']) for e in upstream['files']]).encode()).hexdigest()
    ratio = root / 'data/ratios/intersection_01_ratio_pixel2meter.txt'
    ratio.parent.mkdir(parents=True)
    ratio.write_text('20.0\n')
    upstream['files'].append({'path': str(ratio.relative_to(root)), 'sha': blob_hash(ratio.read_bytes()),
                             'sha256': sha256(ratio), 'size': ratio.stat().st_size})
    result = build_dut(root, out, upstream)
    manifest_path, report_path = tmp_path / 'manifest.json', tmp_path / 'build.json'
    manifest_path.write_text(json.dumps(upstream))
    report_path.write_text(json.dumps(result))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    config = tmp_path / 'configs/m3w_independent_experiment.draft.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'records': {}}))
    evidence = module.audit(root, out, manifest_path, report_path)
    assert evidence['all_raw_rows_verified'] == 780
    assert evidence['tracks'] == 6 and evidence['physical_scene_count'] == 2
    assert len(evidence['exact_full_track_duplicates_within_dut']) == (2 if duplicate_within_clip else 3)
    assert evidence['quarantine_recordings'] == (['dut_intersection_01'] if duplicate_within_clip else [])
    assert not evidence['training_run']
    assert json.loads(json.dumps(evidence))['code_sha256'] == sha256(module.Path(module.__file__))
