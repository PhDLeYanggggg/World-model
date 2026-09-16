"""Raw CITR clip conversion for causal diagnostic access, not split approval."""
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


PHYSICAL_SCENE = 'citr_osu_parking_lot'
ROOT = Path(__file__).resolve().parents[2]


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def parse_raw_object(text, filename):
    match = re.fullmatch(r'([pv])([0-9]+)\.csv', filename)
    if not match:
        raise ValueError('Expected one raw p<ID>/v<ID> object file')
    prefix, digits = match.groups()
    raw_id = int(digits)
    kind, code, columns = ('pedestrian', 0, ('x', 'y')) if prefix == 'p' else ('vehicle', 1, ('x_c', 'y_c'))
    agent_id = 2 * raw_id + code
    if raw_id < 0 or agent_id >= 2**53:
        raise ValueError('Agent identity outside exactly representable integer range')
    reader = csv.DictReader(io.StringIO(text))
    expected = {'frame', 'id', 'type', *columns}
    if prefix == 'v':
        expected |= {'x_1', 'y_1', 'x_2', 'y_2'}
    if reader.fieldnames is None or set(reader.fieldnames) != expected or len(reader.fieldnames) != len(expected):
        raise ValueError('Exact raw-object schema required; filtered positions/velocities are not accepted')
    points = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed raw CSV row')
        frame, identity = float(row['frame']), float(row['id'])
        if (not np.isfinite([frame, identity]).all() or frame != int(frame) or frame < 0 or frame >= 2**53
                or identity != raw_id or row['type'] != ('ped' if prefix == 'p' else 'veh')):
            raise ValueError('Raw frame, object ID or type disagrees with filename')
        points.append([frame, agent_id, float(row[columns[0]]), float(row[columns[1]])])
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError('Nonempty finite strictly ordered raw trajectory required')
    return points, {'agent_id': agent_id, 'source_id': raw_id, 'agent_type': kind,
                    'coordinate_columns': list(columns), 'source_row_count': len(points)}


def source_snapshot(source_root, upstream_report):
    source_root = Path(source_root).resolve()
    source_dir = source_root / 'data/trajectories'
    files, manifest = [], []
    for path in sorted(source_dir.rglob('*.csv')):
        if path.is_symlink() or not path.resolve().is_relative_to(source_root):
            raise ValueError('Raw source symlink/path escape refused')
        relative = str(path.relative_to(source_root))
        data = path.read_bytes()
        files.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        manifest.append((relative, blob_hash(data)))
    digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    if (upstream_report.get('status') != 'verified' or not files
            or upstream_report.get('raw_local_object_files') != len(files)
            or upstream_report.get('raw_path_and_git_blob_manifest_sha256') != digest):
        raise ValueError('Raw file set/bytes differ from the previously verified author snapshot')
    if any(upstream_report.get(k) != 0 for k in ('missing_local_count', 'extra_local_count', 'content_mismatch_count')):
        raise ValueError('Unresolved upstream raw file discrepancy')
    return files, digest


def clip_points(source_root, files):
    arrays, row_sources, agents = [], [], {}
    for file_index, entry in enumerate(files):
        path = Path(source_root) / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Source changed while converting')
        points, agent = parse_raw_object(path.read_text(), path.name)
        key = str(agent['agent_id'])
        if key in agents:
            raise ValueError('One namespaced agent occurs in multiple files in a clip')
        agents[key] = {**agent, 'source_file_index': file_index}
        arrays.append(points)
        row_sources.append(np.column_stack([np.full(len(points), file_index), np.arange(2, len(points)+2)]))
    points, mapping = np.concatenate(arrays), np.concatenate(row_sources).astype(np.int64)
    order = np.lexsort((points[:, 0], points[:, 1]))
    points, mapping = points[order], mapping[order]
    return points, mapping, agents


class CITRRecordingWindows(RecordingWindows):
    """Typed scene identity is contextual metadata; no track-end facts enter inputs."""
    def __init__(self, directory, max_neighbors=8):
        super().__init__(directory, max_neighbors=max_neighbors)
        if self.metadata.get('dataset') != 'CITR' or self.metadata.get('physical_scene') != PHYSICAL_SCENE:
            raise ValueError('Expected the controlled CITR parking-lot recording')
        self.agent_table = self.metadata['agent_table']
        if set(self.agent_table) != {str(int(a)) for a in np.unique(self.points[:, 1])}:
            raise ValueError('Typed agent table does not cover stored positions')
        for key, item in self.agent_table.items():
            code = {'pedestrian': 0, 'vehicle': 1}.get(item['agent_type'])
            if code is None or int(key) != item['agent_id'] or int(key) != 2*item['source_id']+code:
                raise ValueError('Agent namespace/type mismatch')
        mapping = self.directory / 'source_rows.npy'
        if sha256(mapping) != self.metadata['artifacts']['source_rows.npy']['sha256']:
            raise ValueError('Raw row provenance changed')
        self.source_rows = np.load(mapping, mmap_mode='r', allow_pickle=False)
        if self.source_rows.shape != (len(self.points), 2):
            raise ValueError('Raw row provenance shape mismatch')

    def get_scene_inputs(self, frame_id, horizon_raw, history_steps=8):
        scene = super().get_scene_inputs(frame_id, horizon_raw, history_steps=history_steps)
        for agent in scene['agents']:
            agent['agent_type'] = self.agent_table[str(agent['agent_id'])]['agent_type']
        return scene


def verify_clip(directory, files, expected_metadata_hash=None):
    directory = Path(directory)
    if expected_metadata_hash is not None and sha256(directory / 'metadata.json') != expected_metadata_hash:
        raise ValueError('Converted metadata changed')
    reader = CITRRecordingWindows(directory)
    if reader.metadata['files'] != files:
        raise ValueError('Converted source identity differs')
    return reader


def verify_source_rows(reader, source_root):
    checked = 0
    for file_index, entry in enumerate(reader.metadata['files']):
        path = Path(source_root) / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Raw source identity changed')
        expected, _ = parse_raw_object(path.read_text(), path.name)
        rows = np.flatnonzero(reader.source_rows[:, 0] == file_index)
        np.testing.assert_array_equal(reader.points[rows], expected)
        np.testing.assert_array_equal(reader.source_rows[rows, 1], np.arange(2, len(expected)+2))
        checked += len(rows)
    if checked != len(reader.points):
        raise ValueError('Some converted rows do not trace to a raw object file')
    return checked


def build_citr(source_root, output_dir, upstream_report, *, resume=False, progress=None):
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    if output_dir == source_root or output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError('Derived output and raw source trees must remain separate')
    files, source_hash = source_snapshot(source_root, upstream_report)
    identity = {'source_root': str(source_root), 'source_manifest_sha256': source_hash,
                'upstream_commit': upstream_report['upstream_commit'], 'files': files,
                'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                    (Path(__file__), ROOT / 'src/data_unification/m3w_causal_recordings.py')},
                'history_steps': 8, 'future_observation_steps': 12,
                'physical_scene': PHYSICAL_SCENE, 'role': 'diagnostic_only'}
    identity_path = output_dir / 'run_identity.json'
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError('Conversion identity changed; do not overwrite an existing cache')
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    groups = defaultdict(list)
    for entry in files:
        groups[str(Path(entry['path']).parent.relative_to('data/trajectories'))].append(entry)
    began = time.monotonic()
    summaries, reused, source_rows_verified = [], 0, 0
    for clip, entries in sorted(groups.items()):
        if any(not re.fullmatch('[a-zA-Z0-9_]+', part) for part in Path(clip).parts):
            raise ValueError('Unsafe clip identity')
        name = 'citr_' + clip.replace('/', '__')
        directory = output_dir / name
        receipt = directory / 'completion.json'
        if directory.is_symlink():
            raise ValueError('Converted clip symlink refused')
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved.get('run_identity_sha256') != identity_hash:
                raise ValueError('Completion receipt belongs to another conversion')
            reader = verify_clip(directory, entries, saved['metadata_sha256'])
            metadata = reader.metadata
            reused += 1
        else:
            if directory.exists():
                raise ValueError('Completed clip without a receipt; review rather than silently overwrite')
            temporary = output_dir / (name + '.partial')
            if temporary.exists():
                marker = temporary / 'owner.json'
                if (temporary.is_symlink() or not marker.is_file()
                        or json.loads(marker.read_text()) != {'run_identity_sha256': identity_hash, 'clip': name}):
                    raise ValueError('Unowned partial directory; review rather than remove it')
                # Remove only an unfinished directory bound to this exact run.
                shutil.rmtree(temporary)
            temporary.mkdir()
            (temporary / 'owner.json').write_text(json.dumps({'run_identity_sha256': identity_hash, 'clip': name}))
            points, row_sources, agents = clip_points(source_root, entries)
            record = {'id': name, 'dataset': 'CITR', 'source_clip': clip, 'physical_scene': PHYSICAL_SCENE,
                'files': entries, 'agent_table': agents, 'agent_types': dict(Counter(a['agent_type'] for a in agents.values())),
                'historical_use': 'unknown', 'source_conditions_review': 'pending_not_formal_use_approval',
                'source_declared_unit': 'meter_in_author_description_not_independently_verified',
                'coordinate_conversion_applied': False, 'scale_correspondence_verified': False,
                'fps': None, 'effective_seconds': 'unknown', 'frame_mapping': 'author_declared_clip_frame_id',
                'goals_constructed': False, 'upstream_commit': upstream_report['upstream_commit']}
            metadata = write_recording(temporary, points, record)
            if metadata['points'] != len(points) or metadata['exact_duplicate_point_rows_removed']:
                raise ValueError('Unexpected source row loss during conversion')
            np.save(temporary / 'source_rows.npy', row_sources, allow_pickle=False)
            path = temporary / 'source_rows.npy'
            metadata['artifacts']['source_rows.npy'] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
            (temporary / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            reader = verify_clip(temporary, entries)
            np.testing.assert_array_equal(reader.points, points)
            np.testing.assert_array_equal(reader.source_rows, row_sources)
            (temporary / 'completion.json').write_text(json.dumps({
                'metadata_sha256': sha256(temporary / 'metadata.json'),
                'run_identity_sha256': identity_hash})+'\n')
            # The arrays, metadata and completion receipt become visible together.
            os.replace(temporary, directory)
        source_rows_verified += verify_source_rows(reader, source_root)
        summaries.append({k: metadata[k] for k in ('id', 'source_clip', 'physical_scene', 'points', 'agents', 'agent_types',
                        'raw_exact_windows', 'observation_step_windows', 'track_length_quantiles')})
        if progress:
            progress(name, len(summaries), len(groups))
    if source_snapshot(source_root, upstream_report)[1] != source_hash:
        raise ValueError('Raw sources changed during build')
    report = {'result_source': 'cached_verified' if reused == len(groups) else 'fresh_run',
              'source_status': 'cached_verified_author_git_blob_manifest', 'source_manifest_sha256': source_hash,
              'upstream_commit': upstream_report['upstream_commit'], 'recordings': summaries,
              'recording_count': len(summaries), 'physical_scene_groups': [PHYSICAL_SCENE],
              'controlled_clips_not_independent_sites': True, 'reused_recordings': reused,
              'points': sum(r['points'] for r in summaries), 'agents': sum(r['agents'] for r in summaries),
              'agent_types': dict(sum((Counter(r['agent_types']) for r in summaries), Counter())),
              'raw_exact_windows': {str(h): sum(r['raw_exact_windows'][str(h)] for r in summaries) for h in (10, 25, 50, 100)},
              'observation_step_windows': sum(r['observation_step_windows'] for r in summaries),
              'conversion_seconds': time.monotonic()-began,
              'cache_bytes': sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file()),
              'raw_rows_preserved': True, 'source_immutable': True, 'raw_row_mapping_built': True,
              'source_rows_individually_verified': source_rows_verified,
              'coordinate_claim': 'dataset_local_unverified', 'time_claim': 'raw_frames_or_steps_not_seconds',
              'role': 'diagnostic_only', 'data_use_approval': False, 'official_split_assigned': False,
              'independent_confirmation': False, 'training_run': False, 'prediction_accuracy_evaluated': False,
              'stage5c_executed': False, 'smc_enabled': False, 'run_identity_sha256': sha256(identity_path)}
    return report


def check_past_only(cache_root, report):
    """Mutate all post-current positions; compare inputs without calling label APIs."""
    checks, agents_checked, types = [], 0, Counter()
    for record in report['recordings']:
        reader = CITRRecordingWindows(Path(cache_root) / record['id'])
        ids = np.flatnonzero((reader.index['protocol'] == PROTOCOL_RAW) & (reader.index['horizon_raw'] == 50))
        if not len(ids):
            checks.append({'recording': record['id'], 'status': 'not_run_no_exact_raw50_window'})
            continue
        item = int(ids[len(ids)//2])
        frame = reader.identity(item)['frame_id']
        before = reader.get_scene_inputs(frame, 50)
        changed = np.array(reader.points)
        changed[changed[:, 0] > frame, 2:] += 10000.
        reader.points = changed
        after = reader.get_scene_inputs(frame, 50)
        if [a['agent_id'] for a in before['agents']] != [a['agent_id'] for a in after['agents']]:
            raise ValueError('Future corruption changed observed scene membership')
        for left, right in zip(before['agents'], after['agents']):
            if left['agent_type'] != right['agent_type']:
                raise ValueError('Future corruption changed observable agent type')
            for key in left['inputs']:
                np.testing.assert_array_equal(left['inputs'][key], right['inputs'][key])
            for key in left['coordinate_transform']:
                np.testing.assert_array_equal(left['coordinate_transform'][key], right['coordinate_transform'][key])
            types[left['agent_type']] += 1
        agents_checked += len(before['agents'])
        checks.append({'recording': record['id'], 'frame': frame, 'agents': len(before['agents']), 'status': 'inputs_unchanged'})
    return {'result_source': 'fresh_run_causal_input_mutation_check', 'recording_checks': checks,
            'query_sampling': 'one_index_eligible_raw50_frame_per_clip_not_a_formal_evaluation_population',
            'agent_queries_checked': agents_checked, 'agent_types_checked': dict(types), 'future_label_api_calls': 0,
            'predictions_or_accuracy_computed': False, 'in_memory_mutation_only_raw_and_cache_unchanged': True}
