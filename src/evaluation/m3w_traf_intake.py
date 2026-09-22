"""TRAF raw-annotation screening, without geometry conversion or role assignment.

The local README's xyxy interpretation conflicts with many records. Frame/ID
support can still be counted, but cannot establish calibrated motion geometry.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from src.evaluation.m3w_external_source_audit import HISTORIES, HORIZONS, quantiles


KNOWN_CLASSES = {'ped', 'cycle', 'scooter', 'bike', 'rick', 'rickshaw',
                 'car', 'bus', 'truck', 'others'}


def require_geometry_conversion():
    raise ValueError('TRAF box convention unresolved; geometry conversion is not admitted')


def _natural(value):
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError('Expected nonnegative integer frame/count')
    return int(value)


def parse_frame(line):
    fields = next(csv.reader([line]))
    fields = [s.strip() for s in fields]
    if len(fields) < 2:
        raise ValueError('Missing frame/count fields')
    frame, count = map(_natural, fields[:2])
    if len(fields) != 2 + 5 * count:
        raise ValueError('Declared agent count differs from complete record fields')
    agents, seen = [], set()
    for start in range(2, len(fields), 5):
        box = tuple(map(float, fields[start:start + 4]))
        if not all(math.isfinite(v) for v in box):
            raise ValueError('Nonfinite box field')
        agent = fields[start + 4]
        if not agent or not agent.isprintable():
            raise ValueError('Empty or invalid agent identity')
        if agent in seen:
            raise ValueError('Duplicate agent identity within frame')
        match = re.fullmatch(r'([A-Za-z]+)([0-9]+)', agent)
        seen.add(agent)
        agents.append((agent, match.group(1).lower() if match else 'untyped_id', box))
    return frame, agents


def frame_support(frames):
    raw = np.asarray(frames)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0):
        raise ValueError('Frame identities must be a one-dimensional nonnegative integer sequence')
    values = np.sort(raw.astype(np.int64))
    if not len(values) or np.any(np.diff(values) <= 0):
        raise ValueError('Track must have nonempty unique increasing frames')
    result = {'points': len(values), 'frame_min': int(values[0]), 'frame_max': int(values[-1]),
              'discontinuous_edges_at_stride1': int(np.sum(np.diff(values) != 1))}
    for step in (1, 12):
        runs = []
        for phase in range(step):
            f = values[values % step == phase]
            if len(f):
                cuts = np.r_[0, np.flatnonzero(np.diff(f) != step) + 1, len(f)]
                runs.extend(np.diff(cuts).tolist())
        def windows(history, future):
            return sum(max(n - history - future + 1, 0) for n in runs)
        result[f'stride{step}'] = {
            'history_only': {str(k): windows(k, 0) for k in HISTORIES},
            'history_and_12step_future': {str(k): windows(k, 12) for k in HISTORIES}}
        if step == 1:
            result['history8_full_raw_horizon'] = {str(h): windows(8, h) for h in HORIZONS}
    return result


def _sum_support(tracks):
    out = {'track_count': len(tracks), 'point_count': sum(t['points'] for t in tracks),
           'track_length': quantiles([t['points'] for t in tracks]),
           'discontinuous_edges_at_stride1': sum(t['discontinuous_edges_at_stride1'] for t in tracks)}
    for step in (1, 12):
        out[f'stride{step}'] = {kind: {str(k): sum(t[f'stride{step}'][kind][str(k)] for t in tracks)
                                      for k in HISTORIES}
                               for kind in ('history_only', 'history_and_12step_future')}
    out['history8_full_raw_horizon'] = {str(h): sum(t['history8_full_raw_horizon'][str(h)] for t in tracks)
                                       for h in HORIZONS}
    return out


def audit_recording(path: Path):
    content = path.read_bytes()
    groups, kinds, errors, checks = defaultdict(list), {}, [], Counter()
    previous, frame_count, agent_counts = None, 0, []
    for number, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
        try:
            frame, agents = parse_frame(line)
            if previous is not None and frame <= previous:
                errors.append({'line': number, 'kind': 'nonincreasing_frame'})
                continue
            previous = frame
            frame_count += 1
            agent_counts.append(len(agents))
            for agent, kind, box in agents:
                groups[agent].append((frame, *box))
                kinds[agent] = kind
                x, y, a, b = box
                checks['boxes'] += 1
                checks['incompatible_with_xyxy'] += int(a <= x or b <= y)
                checks['positive_third_fourth_values'] += int(a > 0 and b > 0)
                checks['negative_fields'] += int(min(box) < 0)
        except (ValueError, csv.Error) as exc:
            errors.append({'line': number, 'kind': 'parse_failure', 'message': str(exc)})
    if frame_count == 0:
        errors.append({'line': None, 'kind': 'empty_or_unparseable_recording'})
    tracks, fingerprints = [], []
    local_hashes = defaultdict(list)
    for agent, rows in sorted(groups.items()):
        stat = frame_support([r[0] for r in rows])
        stat.update(agent_id=agent, agent_class=kinds[agent])
        tracks.append(stat)
        absolute = np.asarray(rows, dtype='<f8')
        digest = hashlib.sha256(absolute.tobytes()).hexdigest()
        local_hashes[digest].append(agent)
        relative = absolute.copy()
        relative[:, 0] -= relative[0, 0]
        fingerprints.append({'recording': path.stem.removesuffix('_gt'), 'agent': agent,
                             'points': len(rows), 'relative_frame_box_sha256':
                             hashlib.sha256(relative.tobytes()).hexdigest()})
    duplicates = [v for v in local_hashes.values() if len(v) > 1]
    unknown = sorted(set(kinds.values()) - KNOWN_CLASSES)
    result = {
        'recording': path.stem.removesuffix('_gt'), 'file': path.name,
        'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
        'input_lines': len(content.splitlines()), 'parsed_frames': frame_count,
        'frame_error_count': len(errors),
        'frame_error_counts_by_reason': dict(Counter(e.get('message', e['kind']) for e in errors)),
        'frame_errors': errors[:20], 'frame_error_examples_truncated': len(errors) > 20,
        'unknown_classes': unknown,
        'duplicate_tracks_within_recording': duplicates,
        'quality_status': 'quarantined' if errors or unknown or duplicates else 'structure_screen_only',
        'eligible_for_forecast_use': False,
        'agent_count_per_parsed_frame': quantiles(agent_counts),
        'bbox_checks': dict(checks), 'counts_before_recording_quarantine': _sum_support(tracks),
        'by_class_before_recording_quarantine': {kind: _sum_support([t for t in tracks if t['agent_class'] == kind])
                                                for kind in sorted(set(kinds.values()))},
    }
    return result, fingerprints


def audit_directory(root: Path):
    paths = sorted(root.glob('TRAF*_gt.txt'))
    if not paths:
        raise ValueError('No TRAF annotation files')
    recordings, hashes, track_hashes, families = [], defaultdict(list), defaultdict(list), defaultdict(list)
    for path in paths:
        result, fingerprints = audit_recording(path)
        recordings.append(result)
        hashes[result['sha256']].append(path.name)
        family = re.fullmatch(r'TRAF([0-9]+)(?:_[0-9]+)?_gt.txt', path.name)
        families[family.group(1) if family else path.stem].append(path.name)
        for item in fingerprints:
            track_hashes[item['relative_frame_box_sha256']].append(item)
    checks, by_class = Counter(), defaultdict(Counter)
    for item in recordings:
        checks.update(item['bbox_checks'])
        for kind, values in item['by_class_before_recording_quarantine'].items():
            counts = by_class[kind]
            counts['tracks'] += values['track_count']
            counts['points'] += values['point_count']
            for step in (1, 12):
                counts[f'obs8_pred12_stride{step}'] += values[f'stride{step}']['history_and_12step_future']['8']
    return {
        'schema_version': 1, 'result_source': 'fresh_run',
        'scope': 'raw_structural_availability_not_geometry_conversion_or_forecasting',
        'annotation_file_count': len(paths), 'filename_family_count': len(families),
        'filename_families_not_sites': dict(families),
        'independent_physical_site_count': None,
        'physical_scene_mapping_status': 'unknown_filename_families_are_not_sites',
        'recordings': recordings, 'class_counts_before_quarantine': dict(by_class),
        'bbox_checks': dict(checks),
        'exact_file_duplicate_groups': [v for v in hashes.values() if len(v) > 1],
        'exact_relative_frame_box_track_duplicate_groups': [v for v in track_hashes.values() if len(v) > 1],
        'quarantined_recordings': [r['recording'] for r in recordings if r['quality_status'] == 'quarantined'],
        'geometry_interpretation': 'unresolved_no_center_or_footpoint_conversion',
        'coordinate_unit': 'raw_image_box_fields_pixels_not_verified_ground_positions',
        'time_status': 'conflicting_source_fps_no_seconds_claim',
        'history_and_future_support': 'overlapping_all_phase_availability_not_independent_samples_or_selected_protocol',
        'source_use_terms': 'unresolved_dataset_specific_not_inferred_from_toolkit_license',
        'historical_exposure': 'not_cleared', 'admitted_recordings': 0,
        'split_assigned': False, 'new_training': False, 'new_accuracy_evaluation': False,
        'stage5c_executed': False, 'smc_enabled': False,
    }


def render_report(result):
    checks = result['bbox_checks']
    lines = ['# TRAF Raw Intake Audit', '', '## Material Passport', '',
        'Fresh raw-file structural screening. No center/footpoint conversion, model fitting,',
        'forecast scoring, split assignment, goal construction, or metric/seconds claim.',
        'These overlapping availability counts are not independent samples or an approved',
        '8-to-12 evaluation protocol. Stride 12 covers all phases for inventory only.', '',
        f"Files: {result['annotation_file_count']}; filename families: {result['filename_family_count']}.",
        'Independent physical sites: unknown. File families must not become scene IDs.',
        f"Quarantined recordings: {len(result['quarantined_recordings'])}; admissions: 0.", '',
        '## Geometry Conflict', '',
        f"Of {checks.get('boxes', 0):,} parsed boxes, {checks.get('incompatible_with_xyxy', 0):,} violate",
        'the local README xyxy ordering. Positive third/fourth fields are compatible with',
        'width/height but do not establish that convention. Conversion remains refused.',
        'The local dataset README states 20 FPS; the toolkit table states 10 FPS.',
        'Neither is validated against this release/video metadata.', '',
        '## Availability Before Recording Quarantine', '',
        '| Class | Tracks | Points | 8-to-12 stride 1 | 8-to-12 stride 12 |',
        '|---|---:|---:|---:|---:|']
    for kind, c in sorted(result['class_counts_before_quarantine'].items()):
        lines.append(f"| {kind} | {c['tracks']} | {c['points']} | {c['obs8_pred12_stride1']} | {c['obs8_pred12_stride12']} |")
    lines += ['', '## Per Recording', '', '| Recording | Parsed frames | Tracks | Pedestrian tracks | Quality |',
              '|---|---:|---:|---:|---|']
    for r in result['recordings']:
        peds = r['by_class_before_recording_quarantine'].get('ped', {}).get('track_count', 0)
        lines.append(f"| {r['recording']} | {r['parsed_frames']} | {r['counts_before_recording_quarantine']['track_count']} | {peds} | {r['quality_status']} |")
    lines += ['', '## Remaining Admission Requirements', '',
        'Resolve the annotation convention, source-specific terms, physical-site grouping,',
        'camera motion/perspective and historical exposure before any role assignment.',
        'Resolve quarantined IDs/rows without silent deduplication. Exact-box matches are',
        'a limited duplicate screen, not a complete recording-overlap audit.',
        'Source field parsing does not establish causal availability of annotation geometry.',
        'No traffic-domain result is substituted for pedestrian top-down generalization.', '']
    return '\n'.join(lines)
