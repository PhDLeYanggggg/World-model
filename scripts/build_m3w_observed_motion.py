"""Extract fixed past optical-flow summaries on the previously approved fit cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.data_unification.m3w_causal_recordings import causal_coordinate_transform
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_past_video_alignment import native_to_image_xy
from src.world_model.m3w_masked_history_images import MaskedHistoryImageStore
from src.world_model.m3w_offline_visual_data import verify_registration, json_write
from src.world_model.m3w_observed_motion import pair_motion, normalized_motion, validate_past_join, image_to_native


def load_registration(path):
    reg = json.loads(path.read_text())
    if reg['role'] != 'fit_only_exploratory' or not reg['bindings']:
        raise ValueError('Bound fit-only registration required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT / name) != digest:
            raise ValueError('Changed dependency: ' + name)
    parent, contract = verify_registration(ROOT, ROOT / reg['input_registration'])
    directory = ROOT / parent['output'] / 'inputs'
    manifest = json.loads((directory / 'data_manifest.json').read_text())
    for name, digest in manifest['arrays'].items():
        if file_digest(directory / name) != digest:
            raise ValueError('Changed source array: ' + name)
    return reg, parent, contract, directory, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise RuntimeError('Native arm64 required')
    reg, parent, contract, source, source_manifest = load_registration(args.registration)
    sys.path.insert(0, str(ROOT / reg['opencv_runtime']))
    import cv2
    if cv2.__version__ != reg['opencv_version']:
        raise ValueError('Different optical-flow implementation')
    cv2.setNumThreads(1)
    output, reports = ROOT / reg['output'] / 'inputs', ROOT / reg['reports']
    identity = dict(registration_sha256=file_digest(args.registration), parent_protocol_sha256=contract.digest,
                    source_manifest_sha256=file_digest(source / 'data_manifest.json'))
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Cache identity changed')
        for name, digest in manifest['arrays'].items():
            if file_digest(output / name) != digest:
                raise ValueError('Changed motion array')
        print(json.dumps({'result_source': 'cached_verified', 'rows': manifest['rows'], 'new_extractions': 0}))
        return
    a = {n: np.load(source / (n + '.npy'), mmap_mode='r') for n in ('rgb', 'coverage', 'image_rows', 'scale')}
    rows = json.loads((source / 'rows.json').read_text())
    count = len(rows)
    motion, quality = np.zeros((count, 7, 10), np.float32), np.zeros((count, 7, 5), np.float32)
    started, offset, global_row = time.monotonic(), 1, 0
    summary = {}
    def heartbeat(value):
        value = dict(pid=os.getpid(), time_unix=time.time(), **value)
        json_write(output.parent / 'heartbeat.json', value)
        print(json.dumps(value), flush=True)
    for rid in parent['recordings']:
        reader, indices = contract.open_recording(rid, purpose='fit')
        points = reader.points
        available = rid != 'ucy_zara03'
        centers = h = None
        if available:
            directory = (ROOT / reader.metadata['files'][0]['path']).parent
            h = np.loadtxt(directory / 'H.txt')
            if rid.startswith('ucy_'):
                store = MaskedHistoryImageStore(ROOT / parent['zara_cache'] / rid,
                                                observation_mode='offline_annotation_diagnostic')
                if not np.array_equal(store.arrays['row_keys'], points[:, :2].astype(np.int64)):
                    raise ValueError('Source row mismatch')
                centers = np.asarray(store.arrays['image_xy'])
                projected, valid = image_to_native(centers, h)
                # Translation is source provenance, not a learned scene feature.
                shift = points[0, 2:] - projected[0]
                if not valid.all() or np.max(np.abs(projected + shift - points[:, 2:])) > 1e-5:
                    raise ValueError('Supplied image/source transform no longer agrees')
            else:
                centers = native_to_image_xy(points[:, 2:], h, projected_axes='row_col')
        pair_cache = {}
        begin_global = global_row
        checkpoint = output / ('pairs_' + rid + '.npz')
        pair_manifest = output / ('pairs_' + rid + '.json')
        if pair_manifest.exists():
            old = json.loads(pair_manifest.read_text())
            if old['identity'] != identity or old['sha256'] != file_digest(checkpoint):
                raise ValueError('Partial source extraction changed')
            with np.load(checkpoint) as cached:
                for i, key in enumerate(cached['keys']):
                    pair_cache[tuple(key)] = (cached['vectors'][i], cached['magnitudes'][i], cached['quality'][i])
        for item in indices:
            index = reader.index[item]
            begin, current = int(index['history_start']), int(index['current_row'])
            history = points[begin:current + 1]
            expected = np.arange(begin, current + 1) + offset if available else np.zeros(8, np.int64)
            if rows[global_row]['recording'] != rid:
                raise ValueError('Recording/query ordering changed')
            validate_past_join(history, a['image_rows'][global_row], rows[global_row], expected)
            transform = causal_coordinate_transform(history[:, 2:], history[:, 0], int(index['horizon_raw']))
            if transform['scale'] != a['scale'][global_row]:
                raise ValueError('Past-only scale mismatch')
            if available:
                for slot, p in enumerate(range(begin, current)):
                    key = (p, p + 1)
                    if key not in pair_cache:
                        first, second = offset + p, offset + p + 1
                        pair_cache[key] = pair_motion(a['rgb'][first], a['rgb'][second],
                            a['coverage'][first], a['coverage'][second], centers[p], centers[p + 1], h, cv2)
                    vectors, magnitudes, support = pair_cache[key]
                    motion[global_row, slot] = normalized_motion(vectors, magnitudes, transform['rotation'], transform['scale'])
                    quality[global_row, slot] = support
            global_row += 1
            if global_row % 500 == 0:
                heartbeat(dict(state='extracting_past_pairs', recording=rid, rows=global_row, pairs=len(pair_cache)))
        if pair_cache and not pair_manifest.exists():
            keys = sorted(pair_cache)
            output.mkdir(parents=True, exist_ok=True)
            np.savez(checkpoint, keys=np.array(keys), vectors=np.stack([pair_cache[k][0] for k in keys]),
                magnitudes=np.stack([pair_cache[k][1] for k in keys]), quality=np.stack([pair_cache[k][2] for k in keys]))
            json_write(pair_manifest, dict(identity=identity, sha256=file_digest(checkpoint)))
        if available:
            offset += len(points)
        summary[rid] = dict(rows=len(indices), unique_observed_pairs=len(pair_cache),
            result_source='fresh_run_past_motion' if available else 'not_run_no_video_explicit_zero_mask',
            quality_mean=quality[begin_global:global_row].mean((0, 1)).tolist())
    if global_row != source_manifest['rows'] or offset != len(a['rgb']) or not np.isfinite(motion).all():
        raise ValueError('Cohort/image alignment or finite feature failure')
    for name, array in [('motion.npy', motion), ('quality.npy', quality)]:
        np.save(output / name, array, allow_pickle=False)
    report = dict(identity=identity, rows=count, records=summary, shape=list(motion.shape),
        arrays={n: file_digest(output / n) for n in ('motion.npy', 'quality.npy')},
        result_source='fresh_run_past_motion_cached_verified_images', seconds=time.monotonic()-started,
        opencv_version=cv2.__version__, cv_threads=1, future_labels_opened=False,
        development_calibration_confirmation_opened=False, missing_rows_removed=False,
        crop_translation_corrected=True, supplied_H_coordinate_proxy_not_metric=True,
        offline_annotations_not_strict_sensor_as_of=True, stage5c_executed=False, smc_enabled=False)
    json_write(manifest_path, report)
    json_write(reports / 'input_report.json', report)
    heartbeat(dict(state='inputs_complete', rows=count, seconds=report['seconds']))


if __name__ == '__main__':
    main()
