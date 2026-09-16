"""Read-only availability audit; no episode conversion, interpolation, or split.

Window counts are overlapping label-availability views, not independent samples
or a decision about the forecasting protocol. Stored coordinates stay native.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np


HISTORIES = (8, 16, 32, 64)
HORIZONS = (10, 25, 50, 100)


@dataclass
class Track:
    recording: str
    agent: str
    kind: str
    frame: np.ndarray
    xy: np.ndarray
    timestamps: np.ndarray | None = None


def _integers(values, name):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f'{name}: nonfinite or noninteger identity')
    return values.astype(np.int64)


def parse_gc(text: str, agent: str) -> Track:
    values = np.loadtxt(io.StringIO(text)).reshape(-1)
    if not len(values) or len(values) % 3:
        raise ValueError('GC must contain complete position/position/frame triplets')
    rows = values.reshape(-1, 3)
    return Track('grand_central_manual_routes', agent, 'pedestrian',
                 _integers(rows[:, 2], 'frame'), rows[:, :2])


def parse_hermes(text: str, recording: str) -> list[Track]:
    rows = np.loadtxt(io.StringIO(text), ndmin=2)
    if rows.shape[1] != 5:
        raise ValueError('HERMES requires ID FRAME X Y Z')
    ids = _integers(rows[:, 0], 'agent')
    frames = _integers(rows[:, 1], 'frame')
    return [Track(recording, str(agent), 'pedestrian', frames[ids == agent],
                  rows[ids == agent, 2:4]) for agent in np.unique(ids)]


def parse_citr(text: str, recording: str) -> list[Track]:
    reader = csv.DictReader(io.StringIO(text))
    names = set(reader.fieldnames or [])
    coord = ('x', 'y') if {'x', 'y'} <= names else ('x_c', 'y_c')
    if not {'frame', 'id', 'type', *coord} <= names:
        raise ValueError('CITR raw object header missing; filtered data not accepted')
    groups = defaultdict(list)
    for row in reader:
        kind = {'ped': 'pedestrian', 'veh': 'vehicle'}.get(row['type'])
        if kind is None:
            raise ValueError('Unknown CITR agent type')
        agent = int(_integers([row['id']], 'agent')[0])
        groups[(kind, agent)].append([float(row['frame']), float(row[coord[0]]), float(row[coord[1]])])
    return [Track(recording, f'{kind}:{agent}', kind, _integers(np.array(rows)[:, 0], 'frame'),
                  np.array(rows)[:, 1:]) for (kind, agent), rows in groups.items()]


def parse_vru(text: str, agent: str, kind: str) -> Track:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ['', 'timestamp', 'x', 'y']:
        raise ValueError('VRU measurement-index header not recognized')
    rows = np.array([[float(r[k]) for k in ('', 'timestamp', 'x', 'y')] for r in reader])
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError('Empty or invalid VRU track')
    return Track('global_recording_identity_unknown', agent, kind,
                 _integers(rows[:, 0], 'measurement index'), rows[:, 2:4], rows[:, 1])


def parse_wildtrack_frame(text: str, frame: int) -> list[tuple[int, int, int, int]]:
    result = []
    for row in json.loads(text):
        agent, position = _integers([row['personID'], row['positionID']], 'grid identity')
        if not 0 <= position < 480 * 1440:
            raise ValueError('Wild-Track position outside published native grid')
        result.append((int(agent), frame, int(position % 480), int(position // 480)))
    return result


def track_stats(track: Track, step: int, raw_frame_identity: bool = True) -> dict:
    if step <= 0 or int(step) != step:
        raise ValueError('Nominal stride must be a positive integer')
    frame = _integers(track.frame, 'frame')
    xy = np.asarray(track.xy, dtype=float)
    if xy.shape != (len(frame), 2) or not len(frame) or not np.isfinite(xy).all():
        raise ValueError('Invalid track positions')
    order = np.argsort(frame, kind='stable')
    input_out_of_order = bool(np.any(np.diff(frame) < 0))
    frame, xy = frame[order], xy[order]
    stamps = None if track.timestamps is None else np.asarray(track.timestamps, dtype=float)[order]
    if stamps is not None and (stamps.shape != frame.shape or not np.isfinite(stamps).all()):
        raise ValueError('Invalid timestamp shape/value')
    repeated = np.flatnonzero(np.diff(frame) == 0)
    for i in repeated:
        if not np.array_equal(xy[i], xy[i + 1]) or (stamps is not None and stamps[i] != stamps[i + 1]):
            raise ValueError('Conflicting positions/timestamps at same agent-frame')
    keep = np.r_[True, np.diff(frame) != 0]
    frame, xy = frame[keep], xy[keep]
    if stamps is not None:
        stamps = stamps[keep]
    delta = np.diff(frame)
    valid_edge = delta == step
    cadence = None
    if stamps is not None:
        delta_t = np.diff(stamps)
        if np.any(delta_t <= 0):
            raise ValueError('Nonincreasing VRU native timestamp')
        if len(delta_t):
            cadence = float(Counter(np.round(delta_t, 6)).most_common(1)[0][0])
            valid_edge &= np.isclose(delta_t, cadence, rtol=0, atol=1e-6)
    cuts = np.r_[0, np.flatnonzero(~valid_edge) + 1, len(frame)]
    runs = np.diff(cuts)

    def windows(history, future):
        return int(np.maximum(runs - history - future + 1, 0).sum())

    geometry = np.column_stack((frame, xy)).astype('<f8').tobytes()
    return {
        'raw_points': len(order), 'unique_points': len(frame),
        'duplicate_points': len(repeated), 'input_out_of_order': input_out_of_order,
        'frame_min': int(frame[0]), 'frame_max': int(frame[-1]),
        'step_distribution': {str(k): v for k, v in Counter(delta.tolist()).items()},
        'discontinuous_edges': int((~valid_edge).sum()), 'contiguous_runs': runs.tolist(),
        'history_only_windows': {str(k): windows(k, 0) for k in HISTORIES},
        'history_and_12step_future_windows': {str(k): windows(k, 12) for k in HISTORIES},
        'history8_exact_raw_horizon_windows':
            {str(h): windows(8, h // step) if h % step == 0 else 0 for h in HORIZONS}
            if raw_frame_identity else None,
        'native_timestamp_start': None if stamps is None else float(stamps[0]),
        'native_timestamp_modal_step': cadence,
        'geometry_sha256': hashlib.sha256(geometry).hexdigest(),
    }


def quantiles(values):
    if not values:
        return None
    return dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
                    map(float, np.quantile(values, [0, .25, .5, .75, .95, 1]))))


SPECS = {
    'GC': {'pattern': 'Annotation/*.txt', 'step': 20, 'raw_frame_identity': True,
           'coordinate_unit': 'native_image_pixels_axis_interpretation_unverified',
           'scene_support': 'One station recording; agent files are not independent scenes.',
           'terms_status': 'Local README: no license issued; exact release terms unresolved.',
           'readiness': 'Parser available; release identity/terms/exposure review before any new use.'},
    'HERMES': {'pattern': 'Corridor-*/*.txt', 'step': 1, 'raw_frame_identity': True,
               'coordinate_unit': 'native_xy_source_claims_centimeters_not_verified',
               'scene_support': 'Controlled corridor trials; shared setting/participants, not independent natural sites.',
               'terms_status': 'Exact local experiment terms unresolved; archive access not verified.',
               'readiness': 'Possible controlled stress/curriculum support, not natural-domain success.'},
    'Wild-Track': {'pattern': 'annotations_positions/*.json', 'step': 5, 'raw_frame_identity': True,
                   'coordinate_unit': 'native_integer_ground_grid_not_meters',
                   'scene_support': 'Seven synchronized cameras share one scene; not seven domains.',
                   'terms_status': 'Original page inspected, dataset-specific terms not established.',
                   'readiness': 'Previous diagnostic intake found; conversion/terms unresolved.'},
    'CITR': {'pattern': 'data/trajectories/**/*.csv', 'step': 1, 'raw_frame_identity': True,
             'coordinate_unit': 'dataset_local_xy_source_claims_meters_not_verified',
             'scene_support': 'Controlled clips at one parking lot; six scenarios are not six independent sites.',
             'terms_status': 'Author repository inspected; local release permissions unresolved.',
             'readiness': 'Native synchronized clips usable by parser; terms/exposure and grouping pending.'},
    'VRU': {'pattern': '**/*.csv', 'step': 1, 'raw_frame_identity': False,
            'coordinate_unit': 'dataset_local_xy_source_claims_meters_not_verified',
            'scene_support': 'Single-object files with local clocks; global recording/neighbor alignment unknown.',
            'terms_status': 'Source page inspected; newer extended release terms cannot be inherited.',
            'readiness': 'Single-agent history possible; synchronized multi-agent use is blocked.'},
}


def audit_source(root: Path, source: str) -> dict:
    start = time.monotonic()
    spec = SPECS[source]
    source_root = root / source
    candidates = sorted(source_root.glob(spec['pattern']))
    excluded = [p for p in candidates if any(part.startswith('.') for part in p.relative_to(source_root).parts)]
    paths = [p for p in candidates if p not in excluded]
    manifest, bytes_to_paths, geometry_counts = [], defaultdict(list), Counter()
    rows, failures, records = [], [], defaultdict(lambda: {'tracks': 0, 'points': 0, 'obs8_pred12': 0})
    raw_type_points, types = Counter(), Counter()
    track_keys = set()
    wild_rows = defaultdict(list)
    total_bytes = 0

    def add(track):
        key = (track.recording, track.agent)
        if key in track_keys:
            raise ValueError('Agent identity repeated across object files; requires identity resolution')
        stats = track_stats(track, spec['step'], spec['raw_frame_identity'])
        track_keys.add(key)
        rows.append(stats)
        geometry_counts[stats['geometry_sha256']] += 1
        types[track.kind] += 1
        raw_type_points[track.kind] += stats['raw_points']
        record = records[track.recording]
        record['tracks'] += 1
        record['points'] += stats['unique_points']
        record['obs8_pred12'] += stats['history_and_12step_future_windows']['8']

    for path in paths:
        data = path.read_bytes()
        relative = str(path.relative_to(source_root))
        digest = hashlib.sha256(data).hexdigest()
        manifest.append((relative, digest))
        bytes_to_paths[digest].append(relative)
        total_bytes += len(data)
        try:
            text = data.decode('utf-8-sig')
            if source == 'GC':
                tracks = [parse_gc(text, path.stem)]
            elif source == 'HERMES':
                tracks = parse_hermes(text, str(path.relative_to(source_root).with_suffix('')))
            elif source == 'CITR':
                tracks = parse_citr(text, str(path.parent.relative_to(source_root / 'data/trajectories')))
            elif source == 'VRU':
                kind = 'pedestrian' if 'pedestrians' in path.parts else 'cyclist' if 'cyclists' in path.parts else None
                if kind is None:
                    raise ValueError('Unknown VRU class directory')
                tracks = [parse_vru(text, relative, kind)]
            else:
                for agent, frame, x, y in parse_wildtrack_frame(text, int(path.stem)):
                    wild_rows[agent].append([frame, x, y])
                tracks = []
            for track in tracks:
                try:
                    add(track)
                except ValueError as exc:
                    failures.append({'file': relative, 'agent': track.agent, 'error': str(exc)})
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            failures.append({'file': relative, 'error': str(exc)})
    for agent, values in wild_rows.items():
        values = np.asarray(values)
        try:
            add(Track('wildtrack_synchronized_scene', str(agent), 'pedestrian', values[:, 0], values[:, 1:]))
        except ValueError as exc:
            failures.append({'file': 'annotation_sequence', 'agent': str(agent), 'error': str(exc)})

    def summed(key):
        return {str(k): sum(row[key][str(k)] for row in rows) for k in rows[0][key]} if rows else {}

    steps = Counter()
    for row in rows:
        steps.update(row['step_distribution'])
    duplicates = [v for v in bytes_to_paths.values() if len(v) > 1]
    return {
        'source': source, 'result_source': 'fresh_run', 'source_metadata': spec,
        'raw_files': len(paths), 'raw_bytes': total_bytes,
        'excluded_hidden_metadata_files': [str(p.relative_to(source_root)) for p in excluded],
        'path_and_byte_manifest_sha256': hashlib.sha256(json.dumps(manifest, separators=(',', ':')).encode()).hexdigest(),
        'byte_duplicate_groups': len(duplicates),
        'byte_duplicate_excess_files': sum(len(v) - 1 for v in duplicates),
        'byte_duplicate_examples': duplicates[:5],
        'exact_geometry_duplicate_excess_tracks': sum(v - 1 for v in geometry_counts.values() if v > 1),
        'duplicates_are_not_additional_independent_samples': True,
        'tracks_parsed': len(rows), 'raw_points_in_parsed_tracks': sum(r['raw_points'] for r in rows),
        'unique_agent_frame_points': sum(r['unique_points'] for r in rows),
        'duplicate_agent_frame_rows_removed': sum(r['duplicate_points'] for r in rows),
        'track_types': dict(types), 'raw_point_types': dict(raw_type_points),
        'recording_units': None if source == 'VRU' else len(records),
        'recording_summaries': dict(records) if source != 'VRU' else None,
        'independent_scene_count_verified': None,
        'prior_predictive_use': 'unknown_not_proven_unused',
        'tracks_out_of_order': sum(r['input_out_of_order'] for r in rows),
        'discontinuous_edges': sum(r['discontinuous_edges'] for r in rows),
        'observed_frame_or_measurement_stride_counts': dict(sorted(steps.items(), key=lambda p: int(p[0]))),
        'track_length_points': quantiles([r['unique_points'] for r in rows]),
        'contiguous_run_length_points': quantiles([v for r in rows for v in r['contiguous_runs']]),
        'history_only_windows': summed('history_only_windows'),
        'history_and_12step_future_windows': summed('history_and_12step_future_windows'),
        'history8_exact_raw_horizon_windows': summed('history8_exact_raw_horizon_windows') if spec['raw_frame_identity'] else None,
        'raw_horizon_status': 'exact_nominal_annotation_stride_no_interpolation' if spec['raw_frame_identity'] else 'not_run_global_frame_mapping_unverified',
        'native_timestamp_modal_step_track_counts': dict(Counter(str(r['native_timestamp_modal_step']) for r in rows if r['native_timestamp_modal_step'] is not None)),
        'tracks_native_timestamp_starts_zero': sum(r['native_timestamp_start'] == 0 for r in rows),
        'parse_or_identity_failures': len(failures), 'failure_examples': failures[:20],
        'status': 'metadata_audited_not_experiment_approved' if paths and not failures else 'partial_or_missing_source_review_required',
        'metric_claim_allowed': False, 'seconds_horizon_claim_allowed': False,
        'official_split_assigned': False, 'training_or_model_eval_run': False,
        'elapsed_seconds': time.monotonic() - start,
    }


def render_report(result: dict) -> str:
    lines = ['# Local External Source Availability Audit', '',
             'Fresh raw-file parsing and identity/availability counting, not a training or forecasting result.',
             'No coordinates were rescaled, no gaps interpolated, no official split assigned. No raw data exported.', '',
             '| Source | Files | Parsed tracks | Unique agent-frame points | Recording units | obs8/pred12 views | Raw t50 views |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in result['sources']:
        raw = item['history8_exact_raw_horizon_windows']
        lines.append(f"| {item['source']} | {item['raw_files']:,} | {item['tracks_parsed']:,} | {item['unique_agent_frame_points']:,} | {item['recording_units']} | {item['history_and_12step_future_windows'].get('8', 0):,} | {raw.get('50', 0) if raw is not None else 'not_run'} |")
    lines += ['', 'A recording unit is not proof of an independent physical scene, participant population, or untouched test source.',
              'All window views overlap. These counts cannot be summed into an independent calibration sample size.',
              'Availability uses the complete track for metadata only; track/future availability must not become inference features.',
              'The proposed obs8/pred12 view is not yet an approved main protocol; raw horizons are not seconds.', '']
    for item in result['sources']:
        spec = item['source_metadata']
        lines += [f"## {item['source']}", '', f"- Result source: `{item['result_source']}`; status: `{item['status']}`.",
                  f"- Scene support: {spec['scene_support']}", f"- Coordinate status: `{spec['coordinate_unit']}`; no verified metric or seconds claim.",
                  f"- Terms: {spec['terms_status']}", f"- Readiness: {spec['readiness']}",
                  f"- Input manifest digest: `{item['path_and_byte_manifest_sha256']}`.",
                  f"- Track lengths (points): `{item['track_length_points']}`.",
                  f"- Continuous-run lengths: `{item['contiguous_run_length_points']}`.",
                  f"- K=8/16/32/64 with 12 exact subsequent observation steps: `{item['history_and_12step_future_windows']}`.",
                  f"- K=8 and exact raw t10/t25/t50/t100: `{item['history8_exact_raw_horizon_windows']}`.",
                  f"- Discontinuous edges: {item['discontinuous_edges']}; conflicting/invalid tracks or files: {item['parse_or_identity_failures']}.",
                  f"- Byte-identical excess files: {item['byte_duplicate_excess_files']}; geometry-identical excess tracks: {item['exact_geometry_duplicate_excess_tracks']} (counts not automatically promoted or treated as independent).", '']
    lines += ['## Limits', '', 'DUT has reference material only in the inspected local directory; no trajectory audit or new download was performed.',
              'Prior predictive exposure remains unknown unless separately established. Wild-Track has an existing diagnostic intake record.',
              'Source conditions and independent-scene grouping must be settled before adding any source to a formal experiment.',
              'See `source_review.md` for original-source links, release mismatches, loader issues and next actions.',
              'No Stage5C execution, SMC, deployment promotion, or new real-world predictive gain.']
    return '\n'.join(lines) + '\n'
