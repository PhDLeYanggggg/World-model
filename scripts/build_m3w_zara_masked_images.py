"""Build diagnostic-only lazy partial-image histories for the approved fit Zara sources."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_zara_media_lineage import read_vsp, trace_rows, image_coordinates, history_support
from scripts.audit_m3w_annotation_clock_geometry import avi_header
from scripts.run_m3w_stationary_start_probe import atomic_json
from src.world_model.m3w_masked_history_images import (
    ARRAYS, MaskedHistoryImageStore, masked_center_patch, validate_history_rows,
)


def verify_registration(path):
    reg = json.loads(path.read_text())
    for source, digest in reg['bindings'].items():
        if file_digest(ROOT / source) != digest:
            raise ValueError('Changed registered dependency: ' + source)
    for linked in reg['ancestor_registrations']:
        ancestor = json.loads((ROOT / linked).read_text())
        for source, digest in ancestor['bindings'].items():
            if file_digest(ROOT / source) != digest:
                raise ValueError('Changed ancestor dependency: ' + source)
    return reg


def summarize_and_replay(directory, expected_history_rows):
    store = MaskedHistoryImageStore(directory, observation_mode='offline_annotation_diagnostic')
    arrays = store.arrays
    if not np.array_equal(arrays['history_rows'], expected_history_rows):
        raise ValueError('Source history support changed')
    count = np.asarray(arrays['coverage']).sum((1, 2), dtype=np.int64)
    size = store.metadata['crop_size']
    history = arrays['history_rows']
    coverage = count[history] / (size * size)
    keys, all_decoded = arrays['row_keys'], arrays['frame_decode_mask']
    future_independent_requests = True
    queries_with_later_controls = 0
    # Every source window traverses the public reader, not only a sampled batch.
    for index in range(len(store)):
        inputs = store.inputs(index)
        future_independent_requests &= bool(np.all(inputs['source_frames'] <= inputs['query_source_frame']))
        queries_with_later_controls += int(not store.provenance(index)['controls_at_or_before_query'])
        if set(inputs) != {'rgb', 'pixel_coverage', 'frame_mask', 'history_native_xy',
                           'relative_native_xy', 'source_frames', 'query_source_frame', 'agent_id'}:
            raise ValueError('Unexpected inference field')
        if not np.isfinite(inputs['rgb']).all() or not np.isfinite(inputs['pixel_coverage']).all():
            raise ValueError('Nonfinite decoded input')
    return {'rows': len(keys), 'source_agents': len(np.unique(keys[:, 1])),
        'complete_past8_windows': len(store), 'decoded_source_rows': int(all_decoded.sum()),
        'missing_decoded_source_rows': int((~all_decoded).sum()),
        'full_crop_rows': int((count == size*size).sum()),
        'partial_crop_rows': int(((count > 0) & (count < size*size)).sum()),
        'zero_coverage_rows': int((count == 0).sum()),
        'all_eight_full_crop_windows': int(np.all(coverage == 1, axis=1).sum()),
        'windows_with_some_partial_crop': int(np.any((coverage > 0) & (coverage < 1), axis=1).sum()),
        'windows_with_zero_coverage_frame': int(np.any(coverage == 0, axis=1).sum()),
        'windows_with_every_frame_observed_at_least_partly': int(np.all(coverage > 0, axis=1).sum()),
        'mean_window_pixel_coverage': float(coverage.mean()),
        'window_mean_coverage_p0_p50_p90_p100': np.quantile(coverage.mean(1), [0, .5, .9, 1]).tolist(),
        'every_history_read_and_checked': True, 'all_requested_indices_at_or_before_query': future_independent_requests,
        'histories_with_annotation_controls_after_query': queries_with_later_controls,
        'strict_online_causality_verified': False,
        'array_size_bytes': sum((directory / (name + '.npy')).stat().st_size for name in ARRAYS),
        'array_sha256': store.metadata['array_sha256']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--report-dir', required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    reg = verify_registration(args.registration)
    parent = ExperimentContract(json.loads((ROOT / reg['parent_protocol']).read_text()), ROOT)
    if parent.digest != reg['parent_protocol_sha256']:
        raise ValueError('Changed scientific parent')
    sys.path.insert(0, str(ROOT / reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Decoder changed')
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments') or not reports.is_relative_to(ROOT):
        raise ValueError('Private arrays must remain in ignored workspace data')
    identity = {'registration_sha256': file_digest(args.registration), 'parent_protocol_sha256': parent.digest}
    if output.exists():
        if not args.resume or json.loads((output / 'run_identity.json').read_text()) != identity:
            raise ValueError('Resume requires unchanged registered run identity')
    else:
        if reports.exists():
            raise ValueError('Do not overwrite prior reports')
        output.mkdir(parents=True)
        atomic_json(output / 'run_identity.json', identity)
    if shutil.disk_usage(output).free < reg['minimum_free_bytes']:
        raise ValueError('Insufficient disk reserve; no silent smaller cohort')
    started, results, reused, fresh = time.monotonic(), {}, [], []
    for rid in reg['recordings']:
        if parent.protocol['assignments'][rid] != 'fit':
            raise ValueError('Fit sources only')
        reader, _ = parent.open_recording(rid, purpose='fit')
        points = reader.points
        source = ROOT / 'external_data/OpenTraj/datasets/UCY' / rid.removeprefix('ucy_')
        trace = trace_rows(points, read_vsp(source / 'annotation.vsp'), 1)
        if not trace['available'].all():
            raise ValueError('Cannot assign every source row')
        history, _ = history_support(points, trace['latest_control_frame'], trace['available'], 1)
        row_keys = points[:, :2].astype(np.int64)
        validate_history_rows(row_keys, trace['source_frame'], history)
        target = output / rid
        if (target / 'metadata.json').exists():
            if not args.resume:
                raise ValueError('Existing recording needs resume')
            results[rid] = summarize_and_replay(target, history)
            reused.append(rid)
            print(json.dumps({'recording': rid, 'result_source': 'cached_verified', **results[rid]}), flush=True)
            continue
        target.mkdir(exist_ok=True)
        header, n, side = avi_header(source / 'video.avi'), len(points), reg['output_size']
        arrays = {'native_xy': points[:, 2:].astype(np.float64),
            'image_xy': image_coordinates(trace['centered_xy'], header['width'], header['height']),
            'row_keys': row_keys, 'source_frame': trace['source_frame'],
            'latest_control_frame': trace['latest_control_frame'], 'history_rows': history.astype(np.int32)}
        for name, value in arrays.items():
            np.save(target / (name + '.npy'), value, allow_pickle=False)
        rgb = np.lib.format.open_memmap(target / 'rgb.npy', mode='w+', dtype=np.uint8, shape=(n, 3, side, side))
        coverage = np.lib.format.open_memmap(target / 'coverage.npy', mode='w+', dtype=np.uint8, shape=(n, side, side))
        decoded = np.lib.format.open_memmap(target / 'frame_decode_mask.npy', mode='w+', dtype=bool, shape=(n,))
        rgb[:] = 0
        coverage[:] = 0
        decoded[:] = False
        by_frame = {}
        for row, frame in enumerate(trace['source_frame']):
            by_frame.setdefault(int(frame), []).append(row)
        max_requested = max(by_frame)
        with av.open(str(source / 'video.avi')) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in by_frame:
                    image = np.asarray(frame.to_image())
                    for row in by_frame[frame_index]:
                        rgb[row], coverage[row] = masked_center_patch(image, arrays['image_xy'][row], reg['crop_size'], side)
                        decoded[row] = True
                if frame_index % reg['heartbeat_frames'] == 0:
                    rgb.flush()
                    coverage.flush()
                    decoded.flush()
                    atomic_json(output / 'heartbeat.json', {'pid': os.getpid(), 'state': 'decode', 'recording': rid,
                        'frame_index': frame_index, 'decoded_rows': int(decoded.sum()),
                        'elapsed_seconds': time.monotonic() - started})
                if frame_index >= max_requested:
                    break
        rgb.flush()
        coverage.flush()
        decoded.flush()
        hashes = {name: file_digest(target / (name + '.npy')) for name in ARRAYS}
        atomic_json(target / 'metadata.json', {**identity, 'recording_id': rid,
            'physical_scene': parent.protocol['records'][rid]['physical_scene'],
            'source_role': 'fit', 'data_role': 'diagnostic_only', 'formal_training_admitted': False,
            'observation_mode': 'offline_annotation_diagnostic',
            'source_control_provenance_separate_from_inference': True,
            'crop_size': reg['crop_size'], 'output_size': side,
            'coverage_definition': 'visible_source_pixel_count_per_nonoverlapping_block',
            'coordinate_unit': 'dataset_local_unverified', 'metric_or_seconds_claim': False,
            'array_sha256': hashes})
        results[rid] = summarize_and_replay(target, history)
        fresh.append(rid)
        print(json.dumps({'recording': rid, 'result_source': 'fresh_run', **results[rid]}), flush=True)
    if reports.exists():
        receipt = json.loads((output / 'completion.json').read_text())
        if file_digest(reports / 'report.json') != receipt['report_sha256']:
            raise ValueError('Existing report changed')
        old = json.loads((reports / 'report.json').read_text())
        if old['records'] != results:
            raise ValueError('Replayed full-cache statistics differ')
        atomic_json(output / 'last_resume_verification.json', {'result_source': 'cached_verified',
            **identity, 'recordings': reused, 'new_conversions': len(fresh), 'all_input_windows_replayed': True,
            'original_report_unchanged': True, 'elapsed_seconds': time.monotonic() - started})
    else:
        reports.mkdir(parents=True)
        report = {'result_source': 'fresh_run_masked_past_inputs_not_training', **identity,
            'records': results, 'fresh_recordings': fresh, 'cached_verified_recordings': reused,
            'elapsed_seconds': time.monotonic() - started, 'observation_definition_pending': True,
            'primary_protocol_changed': False, 'future_label_api_used': False,
            'development_calibration_confirmation_opened': False, 'new_training': False,
            'media_training_admitted': False, 'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False}
        atomic_json(reports / 'report.json', report)
        lines = ['# Masked Zara Past-Image Input Store', '',
            'Fresh diagnostic input construction; no new training, prediction result or formal cohort admission.', '',
            '| Recording | Source rows | Past8 windows | Full / partial / empty crops | Every frame partly observed | Bytes |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for rid, r in results.items():
            lines.append(f'| {rid} | {r["rows"]} | {r["complete_past8_windows"]} | '
                         f'{r["full_crop_rows"]} / {r["partial_crop_rows"]} / {r["zero_coverage_rows"]} | '
                         f'{r["windows_with_every_frame_observed_at_least_partly"]} | {r["array_size_bytes"]} |')
        lines += ['', 'Pixel coverage is retained per block. No recentering, hallucinated pixels or silent row removal.',
            'Each crop is stored once per source row; past windows are index lists, not materialized eight times.',
            'Explicit offline diagnostic mode preserves later-control provenance. Strict control-as-of mode rejects affected queries.',
            'Formal supervised/evaluation roles are rejected pending the scientific observation decision.', '']
        (reports / 'report.md').write_text('\n'.join(lines))
        atomic_json(output / 'completion.json', {'report_sha256': file_digest(reports / 'report.json'), **identity})
    atomic_json(output / 'heartbeat.json', {'pid': os.getpid(), 'state': 'complete',
        'elapsed_seconds': time.monotonic() - started, 'fresh_recordings': fresh, 'cached_verified_recordings': reused})


if __name__ == '__main__':
    main()
