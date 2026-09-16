"""Unfiltered DUT positions with exact raw-row provenance and causal inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from src.data_unification.m3w_causal_recordings import RecordingWindows, write_recording, PROTOCOL_RAW
from src.evaluation.m3w_recording_lineage import sha256


ROOT = Path(__file__).resolve().parents[2]
PATTERN = r'((intersection|roundabout)_([0-9]{2}))_traj_(ped|veh)\.csv'
SITES = {'intersection': 'dut_intersection', 'roundabout': 'dut_shared_space'}


def raw_name(filename):
    match = re.fullmatch(PATTERN, filename)
    if not match or not 1 <= int(match[3]) <= (17 if match[2] == 'intersection' else 11):
        raise ValueError('Expected a named raw DUT pedestrian or vehicle table')
    return match[1], match[2], match[4]


def parse_raw_table(text, filename):
    _, _, kind = raw_name(filename)
    code, columns = (0, ('x', 'y')) if kind == 'ped' else (1, ('x_c', 'y_c'))
    expected = {'id', 'frame', 'label', *columns}
    if kind == 'veh':
        expected |= {f'{axis}_{corner}' for axis in ('x', 'y') for corner in ('fl', 'fr', 'rr', 'rl')}
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or len(reader.fieldnames) != len(expected) or set(reader.fieldnames) != expected:
        raise ValueError('Exact raw schema required; filtered columns/velocities are refused')
    points, agents, previous = [], {}, {}
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('Malformed raw row')
        numeric = {k: float(v) for k, v in row.items() if k != 'label'}
        if not np.isfinite(list(numeric.values())).all() or row['label'] != kind:
            raise ValueError('Nonfinite value or incorrect source label')
        identity, frame = numeric['id'], numeric['frame']
        if (identity != int(identity) or not 0 <= identity < 2**52
                or frame != int(frame) or not 0 <= frame < 2**53):
            raise ValueError('Integer raw agent/frame identity required')
        identity, frame = int(identity), int(frame)
        agent = 2*identity + code
        if frame <= previous.get(agent, -1):
            raise ValueError('Raw per-agent frames must be strictly increasing')
        previous[agent] = frame
        agents[str(agent)] = {'agent_id': agent, 'source_id': identity,
            'agent_type': 'pedestrian' if kind == 'ped' else 'vehicle', 'coordinate_columns': list(columns)}
        points.append([frame, agent, numeric[columns[0]], numeric[columns[1]]])
    if not points:
        raise ValueError('Nonempty raw table required')
    return np.asarray(points, dtype=np.float64), agents


def source_snapshot(source_root, upstream):
    source_root = Path(source_root)
    files = sorted([e for e in upstream.get('files', []) if e['path'].startswith('data/trajectories/')], key=lambda e: e['path'])
    actual = sorted(str(p.relative_to(source_root)) for p in (source_root / 'data/trajectories').rglob('*.csv'))
    if upstream.get('status') != 'verified' or not files or actual != [e['path'] for e in files]:
        raise ValueError('Raw source snapshot file set changed or is unverified')
    groups, manifest = defaultdict(set), []
    for entry in files:
        path = source_root / entry['path']
        if path.parent != source_root / 'data/trajectories' or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Raw source symlink/path escape refused')
        clip, _, kind = raw_name(path.name)
        groups[clip].add(kind)
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if entry['size'] != len(data) or entry['sha256'] != hashlib.sha256(data).hexdigest() or entry['sha'] != blob:
            raise ValueError('Raw source bytes no longer match pinned author snapshot')
        manifest.append((entry['path'], blob))
    digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    if (len(files) != upstream.get('raw_local_csv_files') or digest != upstream.get('raw_path_and_git_blob_manifest_sha256')
            or any(kinds != {'ped', 'veh'} for kinds in groups.values())):
        raise ValueError('Incomplete clip pairs or source snapshot mismatch')
    return files, digest


def clip_points(source_root, files):
    arrays, maps, agents = [], [], {}
    for i, entry in enumerate(files):
        path = Path(source_root) / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Raw source changed during conversion')
        points, table = parse_raw_table(path.read_text(), path.name)
        if agents.keys() & table.keys():
            raise ValueError('Namespaced agent appears in two source files')
        agents.update({key: {**value, 'source_file_index': i} for key, value in table.items()})
        arrays.append(points)
        maps.append(np.column_stack([np.full(len(points), i), np.arange(2, len(points)+2)]))
    points, mapping = np.concatenate(arrays), np.concatenate(maps).astype(np.int64)
    order = np.lexsort((points[:, 0], points[:, 1]))
    return points[order], mapping[order], agents


class DUTRecordingWindows(RecordingWindows):
    def __init__(self, directory, max_neighbors=8):
        super().__init__(directory, max_neighbors=max_neighbors)
        if self.metadata.get('dataset') != 'DUT' or self.metadata.get('physical_scene') not in SITES.values():
            raise ValueError('Expected raw DUT recording with known physical-site group')
        self.agent_table = self.metadata['agent_table']
        if set(self.agent_table) != {str(int(a)) for a in np.unique(self.points[:, 1])}:
            raise ValueError('Typed agent table does not cover the stored positions')
        for key, entry in self.agent_table.items():
            code = {'pedestrian': 0, 'vehicle': 1}.get(entry['agent_type'])
            if code is None or int(key) != entry['agent_id'] or int(key) != 2*entry['source_id']+code:
                raise ValueError('Agent namespace/type mismatch')
        path = self.directory / 'source_rows.npy'
        if sha256(path) != self.metadata['artifacts']['source_rows.npy']['sha256']:
            raise ValueError('Raw row map changed')
        self.source_rows = np.load(path, allow_pickle=False, mmap_mode='r')
        if self.source_rows.shape != (len(self.points), 2) or self.source_rows.dtype != np.int64:
            raise ValueError('Invalid source row map')

    def get_scene_inputs(self, frame_id, horizon_raw, history_steps=8):
        scene = super().get_scene_inputs(frame_id, horizon_raw, history_steps=history_steps)
        for agent in scene['agents']:
            agent['agent_type'] = self.agent_table[str(agent['agent_id'])]['agent_type']
        return scene


def verify_rows(reader, root):
    count = 0
    for i, entry in enumerate(reader.metadata['files']):
        path = Path(root) / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Raw source identity changed')
        expected, _ = parse_raw_table(path.read_text(), path.name)
        rows = np.flatnonzero(reader.source_rows[:, 0] == i)
        line_numbers = reader.source_rows[rows, 1]
        np.testing.assert_array_equal(np.sort(line_numbers), np.arange(2, len(expected)+2))
        np.testing.assert_array_equal(reader.points[rows], expected[line_numbers-2])
        count += len(rows)
    if count != len(reader.points):
        raise ValueError('Unmapped source rows')
    return count


def build_dut(source_root, output_dir, upstream, *, resume=False, progress=None):
    if any(p.is_symlink() for base in (Path(source_root), Path(output_dir)) for p in (base, *base.parents)):
        raise ValueError('Symlinked source/output roots refused')
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    if source_root == output_dir or output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError('Keep source and derived trees separate')
    files, source_hash = source_snapshot(source_root, upstream)
    identity = {'source_root': str(source_root), 'source_manifest_sha256': source_hash, 'files': files,
        'upstream_commit': upstream['upstream_commit'], 'role': 'diagnostic_only', 'history_steps': 8,
        'future_observation_steps': 12, 'code_sha256': {str(p.relative_to(ROOT)): sha256(p)
            for p in (Path(__file__), ROOT / 'src/data_unification/m3w_causal_recordings.py')}}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity_path = output_dir / 'run_identity.json'
    if resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError('Source/code/run identity changed; cannot resume')
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    groups = defaultdict(list)
    for entry in files:
        groups[raw_name(Path(entry['path']).name)[0]].append(entry)
    summaries, reused, checked = [], 0, 0
    began = time.monotonic()
    for clip, entries in sorted(groups.items()):
        name = 'dut_' + clip
        directory = output_dir / name
        if directory.is_symlink():
            raise ValueError('Derived recording symlink refused')
        receipt = directory / 'completion.json'
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved.get('run_identity_sha256') != identity_hash or saved.get('metadata_sha256') != sha256(directory / 'metadata.json'):
                raise ValueError('Derived metadata/completion receipt changed')
            reader = DUTRecordingWindows(directory)
            if reader.metadata['files'] != entries:
                raise ValueError('Derived source identity changed')
            metadata = reader.metadata
            reused += 1
        else:
            if directory.exists():
                raise ValueError('Incomplete destination without receipt; review required')
            temporary = output_dir / (name + '.partial')
            owner = {'run_identity_sha256': identity_hash, 'recording': name}
            if temporary.exists() or temporary.is_symlink():
                marker = temporary / 'owner.json'
                if temporary.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != owner:
                    raise ValueError('Unowned partial directory; will not remove')
                shutil.rmtree(temporary)
            temporary.mkdir()
            (temporary / 'owner.json').write_text(json.dumps(owner))
            points, mapping, agents = clip_points(source_root, entries)
            _, site, _ = raw_name(Path(entries[0]['path']).name)
            record = {'id': name, 'dataset': 'DUT', 'source_clip': clip, 'physical_scene': SITES[site],
                'files': entries, 'agent_table': agents, 'agent_types': dict(Counter(a['agent_type'] for a in agents.values())),
                'historical_predictive_use': 'unknown_not_proven_untouched',
                'source_conditions_review': 'pending_not_formal_use_approval',
                'source_coordinate_evidence': 'raw_pixels_indicated_by_author_filter_division_conflicts_with_general_README_meter_statement',
                'coordinate_conversion_applied': False, 'scale_correspondence_verified': False,
                'fps': None, 'effective_seconds': 'unknown', 'source_declared_fps': 23.98,
                'author_filter_fps_constant': 23.976, 'video_frame_mapping_verified': False,
                'frame_mapping': 'author_declared_clip_frame_id', 'goals_constructed': False,
                'upstream_commit': upstream['upstream_commit']}
            metadata = write_recording(temporary, points, record)
            if metadata['points'] != len(points) or metadata['exact_duplicate_point_rows_removed']:
                raise ValueError('Unexpected raw row loss')
            np.save(temporary / 'source_rows.npy', mapping, allow_pickle=False)
            path = temporary / 'source_rows.npy'
            metadata['artifacts']['source_rows.npy'] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
            (temporary / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            reader = DUTRecordingWindows(temporary)
            np.testing.assert_array_equal(reader.points, points)
            np.testing.assert_array_equal(reader.source_rows, mapping)
            (temporary / 'completion.json').write_text(json.dumps({'run_identity_sha256': identity_hash,
                'metadata_sha256': sha256(temporary / 'metadata.json')})+'\n')
            os.replace(temporary, directory)
        checked += verify_rows(reader, source_root)
        summary = {k: metadata[k] for k in ('id', 'source_clip', 'physical_scene', 'points', 'agents', 'agent_types',
            'raw_exact_windows', 'observation_step_windows', 'track_length_quantiles', 'observation_step_actual_raw_horizons')}
        # Availability is not a decision population: scene queries remain past-only.
        summary['native_frame_stride_counts'] = dict(sorted(Counter(str(int(d)) for d in np.diff(reader.points[:, 0])[
            np.diff(reader.points[:, 1]) == 0]).items()))
        summaries.append(summary)
        if progress:
            progress(name, len(summaries), len(groups))
    if source_snapshot(source_root, upstream)[1] != source_hash:
        raise ValueError('Source changed during build')
    return {'result_source': 'cached_verified' if reused == len(groups) else 'fresh_run',
        'upstream_commit': upstream['upstream_commit'], 'source_manifest_sha256': source_hash,
        'recordings': summaries, 'recording_count': len(summaries), 'reused_recordings': reused,
        'physical_scene_groups': sorted({r['physical_scene'] for r in summaries}),
        'clips_not_independent_sites': True, 'points': sum(r['points'] for r in summaries),
        'agents': sum(r['agents'] for r in summaries),
        'agent_types': dict(sum((Counter(r['agent_types']) for r in summaries), Counter())),
        'raw_exact_windows': {str(h): sum(r['raw_exact_windows'][str(h)] for r in summaries) for h in (10,25,50,100)},
        'observation_step_windows': sum(r['observation_step_windows'] for r in summaries),
        'source_rows_individually_verified': checked, 'conversion_seconds': time.monotonic()-began,
        'cache_bytes': sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file()),
        'coordinate_claim': 'dataset_local_unverified', 'time_claim': 'raw_frames_or_steps_not_seconds',
        'role': 'diagnostic_only', 'official_split_assigned': False, 'data_use_approval': False,
        'independent_confirmation': False, 'training_run': False, 'prediction_accuracy_evaluated': False,
        'stage5c_executed': False, 'smc_enabled': False, 'run_identity_sha256': sha256(identity_path)}


def check_past_only(cache_root, report):
    records, count = [], 0
    for r in report['recordings']:
        reader = DUTRecordingWindows(Path(cache_root) / r['id'])
        ids = np.flatnonzero((reader.index['protocol'] == PROTOCOL_RAW) & (reader.index['horizon_raw'] == 50))
        if not len(ids):
            records.append({'recording': r['id'], 'status': 'not_run_no_raw50_window'})
            continue
        frame = reader.identity(int(ids[len(ids)//2]))['frame_id']
        before = reader.get_scene_inputs(frame, 50)
        mutated = np.array(reader.points)
        mutated[mutated[:, 0] > frame, 2:] += 10000
        reader.points = mutated
        after = reader.get_scene_inputs(frame, 50)
        if [a['agent_id'] for a in before['agents']] != [a['agent_id'] for a in after['agents']]:
            raise ValueError('Future coordinates changed visible scene membership')
        for left, right in zip(before['agents'], after['agents']):
            assert left['agent_type'] == right['agent_type']
            for section in ('inputs', 'coordinate_transform'):
                for key in left[section]:
                    np.testing.assert_array_equal(left[section][key], right[section][key])
        count += len(before['agents'])
        records.append({'recording': r['id'], 'frame': frame, 'agents': len(before['agents']), 'status': 'inputs_unchanged'})
    return {'result_source': 'fresh_run_causal_mutation_check', 'records': records,
        'checks_passed': sum(r['status'] == 'inputs_unchanged' for r in records), 'agent_queries_checked': count,
        'query_sampling': 'one_index_eligible_raw50_frame_per_clip_not_formal_evaluation_population',
        'future_label_api_calls': 0, 'predictions_or_accuracy_computed': False, 'on_disk_positions_unchanged': True}
