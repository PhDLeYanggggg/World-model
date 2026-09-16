"""Source timing and homography traces; never silently relabel experiment units."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_recording_lineage import read_track, sha256


def avi_header(path):
    """Read RIFF metadata without decoding or loading video frames."""
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'AVI ':
            raise ValueError('Expected RIFF AVI source, not a guessed video clock')
        end = min(path.stat().st_size, 8 + struct.unpack('<I', header[4:8])[0])

        def chunks(limit):
            while stream.tell() + 8 <= limit:
                tag, size = struct.unpack('<4sI', stream.read(8))
                start = stream.tell()
                if start + size > limit:
                    raise ValueError('Truncated RIFF chunk')
                if tag == b'avih':
                    if size < 40:
                        raise ValueError('Truncated AVI main header')
                    fields = struct.unpack('<10I', stream.read(40))
                    if not fields[0]:
                        raise ValueError('AVI frame interval unavailable')
                    return {'microseconds_per_frame': fields[0], 'header_fps': 1e6/fields[0],
                            'total_frames': fields[4], 'width': fields[8], 'height': fields[9]}
                if tag == b'LIST' and size >= 4 and stream.read(4) == b'hdrl':
                    found = chunks(start + size)
                    if found:
                        return found
                stream.seek(start + size + size % 2)
            return None
        result = chunks(end)
        if result is None:
            raise ValueError('No AVI main header')
        return result


def project_points(points, matrix):
    p, h = np.asarray(points, float), np.asarray(matrix, float)
    if p.ndim != 2 or p.shape[1] != 2 or h.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError('Finite 2D points and 3x3 homography required')
    out = np.column_stack([p, np.ones(len(p))]) @ h.T
    if np.any(np.abs(out[:, 2]) < 1e-12):
        raise ValueError('Homography projects onto an undefined plane point')
    return out[:, :2] / out[:, 2:]


def align_source_rows(pixel, stored):
    lookups = [{tuple(row[:2]): i for i, row in enumerate(array)} for array in (pixel, stored)]
    if any(len(mapping) != len(array) for mapping, array in zip(lookups, (pixel, stored))):
        raise ValueError('Duplicate frame/agent key cannot be disambiguated')
    pkeys, skeys = (set(m) for m in lookups)
    if skeys - pkeys:
        raise ValueError('Stored row lacks source pixel identity')
    indices = [lookups[0][tuple(row[:2])] for row in stored]
    return pixel[indices], {'matched_stored_rows': len(stored), 'pixel_source_rows': len(pixel),
                           'pixel_only_frame_agent_keys': [list(k) for k in sorted(pkeys-skeys)]}


def clock_status(values, counts, header_fps, documented_interval):
    modal = float(values[int(np.argmax(counts))])
    conflict = documented_interval is not None and not np.isclose(modal/header_fps, documented_interval)
    return {'modal_annotation_frame_step': modal,
        'documented_interval_header_step_conflict': bool(conflict),
        'time_status': ('conflicting_annotation_step_documentation_and_video_header' if conflict else
            'source_clock_consistent_but_frame_mapping_unverified' if documented_interval is not None else
            'video_fps_known_annotation_source_mapping_not_fully_verified'),
        'effective_seconds_verified': False, 'raw50_effective_seconds': None}


def main():
    base = ROOT / 'external_data/OpenTraj/datasets'
    inputs, records = set(), {}
    for dataset, sequence in (('ETH', 'seq_eth'), ('ETH', 'seq_hotel'), ('UCY', 'students03')):
        directory = base / dataset / sequence
        source, video, hfile = (directory / n for n in ('obsmat.txt', 'video.avi', 'H.txt'))
        points, skipped = read_track(source)
        if skipped:
            raise ValueError('Malformed position source')
        intervals = []
        for agent in np.unique(points[:, 1]):
            intervals.extend(np.diff(np.sort(points[points[:, 1] == agent, 0])).tolist())
        values, counts = np.unique(intervals, return_counts=True)
        h = np.loadtxt(hfile)
        header = avi_header(video)
        info_path = directory / 'info.txt'
        timing_statement = (info_path.read_text() if info_path.exists() else (base / dataset / 'README.md').read_text())
        explicit_interval = 'timestep of 0.4 seconds' in timing_statement
        inputs.update((source, video, hfile, info_path if info_path.exists() else base / dataset / 'README.md'))
        documented_interval = .4 if explicit_interval else None
        records[sequence] = {'rows': len(points), 'frame_step_counts': {str(k): int(v) for k, v in zip(values, counts)},
            'video_header': header, 'source_explicit_annotation_interval_seconds': .4 if explicit_interval else None,
            **clock_status(values, counts, header['header_fps'], documented_interval),
            'homography_present': True, 'homography_determinant': float(np.linalg.det(h)),
            'metric_calibration_independently_verified': False, 'source_velocity_columns_used_by_reader': False}
    student = base / 'UCY/students03'
    pixel = np.loadtxt(student / 'obsmat_px.txt')
    world = np.loadtxt(student / 'obsmat.txt')
    aligned_pixel, alignment = align_source_rows(pixel, world)
    traces = {}
    for name in ('H.txt', 'H-old.txt'):
        projected = project_points(aligned_pixel[:, [2, 4]], np.loadtxt(student / name))
        errors = np.linalg.norm(projected-world[:, [2, 4]], axis=1)
        traces[name] = {'max_error_dataset_local': float(errors.max()), 'median_error_dataset_local': float(np.median(errors)),
                        'matches_to_stored_decimal_precision_1e_minus6': bool(errors.max() <= 1e-6)}
    records['students03']['pixel_to_stored_coordinate_replay'] = {
        **alignment, 'supplied_matrix_comparisons': traces, 'physical_scale_proved_by_replay': False}
    inputs.update((student / 'obsmat_px.txt', student / 'px2ground.py', student / 'H-old.txt', base / 'ETH/README.md', base / 'UCY/README.md'))
    result = {'result_source': 'fresh_run_local_source_metadata_and_geometry_replay', 'records': records,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(inputs)},
        'script_sha256': sha256(Path(__file__)), 'current_v6_protocol_changed': False,
        'new_metric_or_seconds_performance_claim': False, 'independent_confirmation': False,
        'annotation_interpolation_sensor_as_of_causality_verified': False,
        'sdd_not_audited_here': True, 'stage5c_executed': False, 'smc_enabled': False}
    out = ROOT / 'outputs/publication_readiness_2026_09/annotation_clock_geometry'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Annotation Clock and Geometry Evidence', '',
        'Fresh local metadata inspection and coordinate replay, not a new model evaluation. The running v6 task and error units remain unchanged.', '',
        '| Recording | Header FPS | Documented interval | Frame-step values | Clock conflict |',
        '| --- | ---: | --- | --- | --- |']
    for name, r in records.items():
        lines.append(f"| {name} | {r['video_header']['header_fps']} | {r['source_explicit_annotation_interval_seconds']} | {list(r['frame_step_counts'])} | {r['documented_interval_header_step_conflict']} |")
    lines += ['', 'ETH and Hotel source info explicitly describes 0.4-second annotation intervals. However ETH obsmat uses six-frame steps while its AVI header declares 25 fps: 6/25 is 0.24, not 0.4. A source/encoding clock mapping is missing; do not select the more favorable interpretation. Hotel ten-frame steps are consistent with its documentation and header, but independent video/annotation synchronization is still not established.',
        'Twelve intervals would be 4.8 seconds only conditional on the documented 0.4-second clock. Eight observed points would span seven intervals (2.8 seconds), not eight. Neither is promoted to a verified benchmark conversion. The current protocol keeps native observation steps and raw-frame offsets; raw50 effective seconds remain unverified.', '',
        f"Students03 has {len(pixel)} pixel rows and {len(world)} stored rows. All stored rows match unique frame/agent pixel keys; {len(alignment['pixel_only_frame_agent_keys'])} pixel-only keys at frame5391 are listed in the JSON rather than silently discarded. Their omission reason is not established here.",
        f"The supplied H-old.txt reproduces stored coordinates with maximum error {traces['H-old.txt']['max_error_dataset_local']:.9g}; H.txt differs by up to {traces['H.txt']['max_error_dataset_local']:.9g} stored units. The current conversion script names H.txt, so that script/matrix pair does not reproduce this obsmat artifact. A future scene-image pipeline must bind the matching matrix explicitly. This establishes numerical lineage for H-old, not physical ground-plane accuracy or metric units.",
        'The supplied px2ground.py uses forward differences for velocity. The experiment reader takes only frame, identity and position columns and recomputes causal finite differences; supplied velocity columns are not model inputs.',
        'UCY source documentation describes interpolation between sparse spline annotations. Past-only reader access does not verify the causal construction of those annotation positions. A strict sensor-as-of claim remains unsupported.',
        'Students01 lacks a locally matched homography/video pair here. Zara03 fitting remains a packaged 20-point population; this audit does not recover its continuous identities. Neither issue is silently repaired or used to redefine the live training experiment.',
        'Homography presence, source-reported meters, verified geometric measurement and true 3D are different claims. No SDD timing or geometry conclusion follows from this audit. No metric/time performance claim, new deployment, Stage5C or SMC.', '']
    (out / 'audit.md').write_text('\n'.join(lines))
    print(json.dumps({'records': records, 'experiment_units_changed': False}))


if __name__ == '__main__':
    main()
