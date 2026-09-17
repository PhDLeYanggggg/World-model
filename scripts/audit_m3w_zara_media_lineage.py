"""Numerically trace Zara supplied annotations before broad visual training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_recording_lineage import read_track
from src.evaluation.m3w_zara_media_lineage import read_vsp, trace_rows, image_coordinates, history_support
from scripts.audit_m3w_annotation_clock_geometry import avi_header, project_points
from scripts.run_m3w_stationary_start_probe import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT / path) != digest:
            raise ValueError('Changed source: ' + path)
    parent = ExperimentContract(json.loads((ROOT / reg['parent_protocol']).read_text()), ROOT)
    if parent.digest != reg['parent_protocol_sha256']:
        raise ValueError('Changed parent experiment')
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments') or not reports.is_relative_to(ROOT):
        raise ValueError('Workspace output required; row arrays must remain ignored')
    if reports.exists():
        raise ValueError('No overwrite of reports')
    output.mkdir(parents=True, exist_ok=False)
    started, result, row_files = time.monotonic(), {}, {}
    for rid in reg['recordings']:
        if parent.protocol['assignments'][rid] != 'fit':
            raise ValueError('Only fit recordings permitted')
        reader, _ = parent.open_recording(rid, purpose='fit')
        directory = ROOT / 'external_data/OpenTraj/datasets/UCY' / rid.removeprefix('ucy_')
        points, skipped = read_track(directory / 'obsmat.txt')
        if skipped or not np.array_equal(reader.points, points):
            raise ValueError('Canonical source/cache rows not identical')
        tracks = read_vsp(directory / 'annotation.vsp')
        header = avi_header(directory / 'video.avi')
        variants, arrays = [], {'stored_frame_agent': points[:, :2]}
        for offset in (0, 1):
            traced = trace_rows(points, tracks, offset)
            available = traced['available']
            centered = traced['centered_xy'][available]
            image = image_coordinates(centered, header['width'], header['height'])
            variants_xy = {'centered_xy': centered, 'image_xy': image, 'image_row_col': image[:, ::-1]}
            for hname in reg['matrices'][rid]:
                matrix = np.loadtxt(directory / hname)
                for convention, coords in variants_xy.items():
                    predicted = project_points(coords, matrix)
                    error = np.linalg.norm(predicted - points[available, 2:], axis=1)
                    variants.append({'matrix': hname, 'coordinate_convention': convention, 'stored_frame_offset': offset,
                        'available_rows': int(available.sum()), 'untraceable_rows': int((~available).sum()),
                        'max_native_error': float(error.max()), 'median_native_error': float(np.median(error)),
                        'rows_error_le_1e_minus5': int((error <= 1e-5).sum()),
                        'all_stored_rows_reproduced': bool(available.all() and np.all(error <= 1e-5))})
            for key in ('source_frame', 'latest_control_frame', 'exact_control', 'available', 'centered_xy'):
                arrays[f'offset{offset}_{key}'] = traced[key]
        matches = [v for v in variants if v['all_stored_rows_reproduced']]
        trace_summary = {}
        for offset in (0, 1):
            traced = trace_rows(points, tracks, offset)
            windows, strict = history_support(points, traced['latest_control_frame'], traced['available'], offset)
            arrays[f'offset{offset}_past_windows'] = windows
            arrays[f'offset{offset}_control_as_of_query'] = strict
            complete20, stationary8, unique_agents = 0, 0, set()
            lookup = {(int(p[0]), int(p[1])) for p in points}
            for history in windows:
                last = points[history[-1]]
                unique_agents.add(int(last[1]))
                stationary8 += int(np.array_equal(points[history, 2:], np.broadcast_to(last[2:], (8, 2))))
                complete20 += int(all((int(last[0] + 10 * j), int(last[1])) in lookup for j in range(1, 13)))
            source_frame = traced['source_frame'][traced['available']]
            latest = traced['latest_control_frame'][traced['available']]
            future_support = latest > source_frame
            trace_summary[str(offset)] = {
                'available_rows': int(traced['available'].sum()), 'untraceable_rows': int((~traced['available']).sum()),
                'rows_using_control_after_own_frame': int(future_support.sum()),
                'latest_control_delay_frames_p50_p90_max': [float(v) for v in np.quantile(latest - source_frame, [.5, .9, 1])],
                'complete_eight_past_windows': len(windows), 'eight_past_agents': len(unique_agents),
                'windows_all_controls_at_or_before_query': int(strict.sum()),
                'windows_with_control_after_query': int((~strict).sum()),
                'fully_stationary_eight_past_windows': stationary8,
                'complete_eight_plus_twelve_label_windows': complete20,
                'strict_count_used_to_filter_model_data': False,
            }
        path = output / (rid + '_lineage.npz')
        np.savez(path, **arrays)
        row_files[rid] = file_digest(path)
        result[rid] = {'stored_rows': len(points), 'stored_agents': len(np.unique(points[:, 1])),
            'raw_splines': len(tracks), 'raw_control_points': sum(len(t) for t in tracks),
            'native_frame_steps': sorted(set(np.concatenate([np.diff(np.sort(points[points[:, 1] == agent, 0]))
                                        for agent in np.unique(points[:, 1])]).astype(int).tolist())),
            'video_header': header, 'physical_scene': parent.protocol['records'][rid]['physical_scene'],
            'matrix_convention_comparisons': variants, 'fully_reproducing_conventions': matches,
            'offset_provenance_counts': trace_summary,
            'past_media_admitted_for_training': False, 'metric_or_seconds_verified': False}
        atomic_json(output / 'heartbeat.json', {'state': 'source_trace', 'pid': os.getpid(),
                    'recording': rid, 'elapsed_seconds': time.monotonic() - started})
        print(json.dumps({'recording': rid, 'matches': matches, 'provenance': trace_summary}), flush=True)
    reports.mkdir(parents=True)
    report = {'result_source': 'fresh_run_fit_source_row_and_annotation_construction_trace',
        'registration_sha256': file_digest(args.registration), 'records': result,
        'elapsed_seconds': time.monotonic() - started, 'row_cache_sha256': row_files,
        'new_training': False, 'old_protocol_modified': False, 'new_deployment': False,
        'stage5c_executed': False, 'smc_enabled': False}
    atomic_json(reports / 'audit.json', report)
    lines = ['# Zara Source and Annotation Construction Audit', '',
             'Fit-only source trace, not new training, media admission or physical calibration.', '',
             '| Source | Rows | Agents | Raw splines | Fully matching conventions | Header FPS |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for rid, r in result.items():
        lines.append(f'| {rid} | {r["stored_rows"]} | {r["stored_agents"]} | {r["raw_splines"]} | '
                     f'{len(r["fully_reproducing_conventions"])} | {r["video_header"]["header_fps"]} |')
    lines += ['', 'All coordinate/frame conventions and later-control provenance counts are retained in audit.json.',
              'Ordinary annotation-timestamp causality and strict sensor-as-of causality are distinct.',
              'No row filtering, metric/seconds claim or change to existing model results follows.', '']
    (reports / 'audit.md').write_text('\n'.join(lines))
    atomic_json(output / 'completion.json', {'report_sha256': file_digest(reports / 'audit.json'),
                'row_cache_sha256': row_files, 'registration_sha256': file_digest(args.registration)})
    atomic_json(output / 'heartbeat.json', {'state': 'complete', 'pid': os.getpid(),
                'elapsed_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
