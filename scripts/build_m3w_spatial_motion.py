"""Native observed crops and fixed resolution/pooling controls on approved fit rows."""
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
from scripts.build_m3w_observed_motion import load_registration
from src.data_unification.m3w_causal_recordings import causal_coordinate_transform
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_past_video_alignment import native_to_image_xy
from src.world_model.m3w_masked_history_images import MaskedHistoryImageStore, masked_center_patch
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_observed_motion import validate_past_join
from src.world_model.m3w_spatial_motion import reduce_patch, dense_pair, normalized_grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    reg, parent, contract, source, receipt = load_registration(args.registration)
    sys.path.insert(0, str(ROOT/reg['opencv_runtime']))
    sys.path.insert(0, str(ROOT/parent['decoder_path']))
    import cv2
    import av
    if cv2.__version__ != reg['opencv_version'] or av.__version__ != parent['decoder_version']:
        raise ValueError('Source decoder or flow runtime changed')
    cv2.setNumThreads(1)
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments'):
        raise ValueError('Private native arrays must stay under ignored data')
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 5*1024**3:
        raise RuntimeError('Insufficient native-cache disk reserve; do not silently shrink cohort')
    identity = dict(registration_sha256=file_digest(args.registration), parent_protocol_sha256=contract.digest,
                    source_manifest_sha256=file_digest(source/'data_manifest.json'))
    manifest_path = output/'inputs/manifest.json'
    if manifest_path.exists():
        report = json.loads(manifest_path.read_text())
        if report['identity'] != identity:
            raise ValueError('Changed input identity')
        for name, digest in report['arrays'].items():
            if file_digest(output/'inputs'/name) != digest:
                raise ValueError('Motion input cache changed')
        for rid, entry in report['records'].items():
            if 'native_arrays' in entry:
                for name, digest in entry['native_arrays'].items():
                    if file_digest(output/'native'/rid/name) != digest:
                        raise ValueError('Native source cache changed')
        print(json.dumps(dict(result_source='cached_verified', rows=report['rows'], new_extractions=0)))
        return
    a = {k: np.load(source/(k+'.npy'), mmap_mode='r') for k in ('rgb','coverage','image_rows','scale')}
    rows = json.loads((source/'rows.json').read_text())
    motion = np.zeros((len(rows),3,7,16,3), np.float32)
    quality = np.zeros((len(rows),2,7,16,2), np.float32)
    started, offset, global_row, summaries = time.monotonic(), 1, 0, {}
    def heartbeat(**value):
        payload = dict(pid=os.getpid(), time_unix=time.time(), **value)
        json_write(output/'heartbeat.json', payload)
        print(json.dumps(payload), flush=True)
    for rid in parent['recordings']:
        reader, indices = contract.open_recording(rid, purpose='fit')
        points = reader.points
        available = rid != 'ucy_zara03'
        first_global, pair_cache, metadata = global_row, {}, None
        if available:
            directory = (ROOT/reader.metadata['files'][0]['path']).parent
            h = np.loadtxt(directory/'H.txt')
            if rid.startswith('ucy_'):
                old = MaskedHistoryImageStore(ROOT/parent['zara_cache']/rid, observation_mode='offline_annotation_diagnostic')
                if not np.array_equal(old.arrays['row_keys'], points[:,:2].astype(np.int64)):
                    raise ValueError('Zara source key mismatch')
                centers = np.asarray(old.arrays['image_xy'])
                frames = np.asarray(old.arrays['source_frame'])
            else:
                centers = native_to_image_xy(points[:,2:], h, projected_axes='row_col')
                frames = points[:,0].astype(np.int64)
            native = output/'native'/rid
            native.mkdir(parents=True, exist_ok=True)
            native_meta = native/'manifest.json'
            if native_meta.exists():
                metadata = json.loads(native_meta.read_text())
                if metadata['identity'] != identity:
                    raise ValueError('Native recording identity changed')
                for name, digest in metadata['arrays'].items():
                    if file_digest(native/name) != digest:
                        raise ValueError('Native array changed')
                rgb = np.load(native/'rgb.npy', mmap_mode='r')
                cov = np.load(native/'coverage.npy', mmap_mode='r')
            else:
                rgb = np.lib.format.open_memmap(native/'rgb.npy', mode='w+', dtype=np.uint8, shape=(len(points),3,96,96))
                cov = np.lib.format.open_memmap(native/'coverage.npy', mode='w+', dtype=np.uint8, shape=(len(points),96,96))
                rgb[:] = 0
                cov[:] = 0
                requests, seen = {}, set()
                for i, frame in enumerate(frames):
                    requests.setdefault(int(frame), []).append(i)
                with av.open(str(directory/'video.avi')) as video:
                    for frame_id, frame in enumerate(video.decode(video=0)):
                        if frame_id in requests:
                            image = np.asarray(frame.to_image())
                            seen.add(frame_id)
                            for i in requests[frame_id]:
                                rgb[i], cov[i] = masked_center_patch(image, centers[i], 96, 96)
                                reduced, counts = reduce_patch(rgb[i], cov[i])
                                if not np.array_equal(reduced, a['rgb'][offset+i]) or not np.array_equal(counts,a['coverage'][offset+i]):
                                    raise ValueError('Native decode no longer reproduces registered old crop')
                        if frame_id % 1000 == 0:
                            heartbeat(state='native_decode', recording=rid, frame=frame_id)
                        if frame_id >= max(requests):
                            break
                if seen != set(requests):
                    raise ValueError('Missing observed frame; do not drop rows')
                rgb.flush()
                cov.flush()
                np.save(native/'image_xy.npy', centers, allow_pickle=False)
                np.save(native/'source_frame.npy', frames, allow_pickle=False)
                np.save(native/'row_keys.npy', points[:,:2].astype(np.int64), allow_pickle=False)
                metadata = dict(identity=identity, all_downsampled_rows_exact=True, rows=len(points),
                    arrays={k:file_digest(native/k) for k in ('rgb.npy','coverage.npy','image_xy.npy','source_frame.npy','row_keys.npy')})
                json_write(native_meta, metadata)
        pair_file, pair_receipt = output/(rid+'_pairs.npz'), output/(rid+'_pairs.json')
        if pair_receipt.exists():
            entry = json.loads(pair_receipt.read_text())
            if entry['identity'] != identity or entry['sha256'] != file_digest(pair_file):
                raise ValueError('Partial flow extraction changed')
            with np.load(pair_file) as saved:
                for i, key in enumerate(saved['keys']):
                    pair_cache[int(key)] = (saved['grid'][i], saved['quality'][i])
        for item in indices:
            index = reader.index[item]
            begin, current = int(index['history_start']), int(index['current_row'])
            history = points[begin:current+1]
            expected = np.arange(begin,current+1)+offset if available else np.zeros(8,np.int64)
            if rows[global_row]['recording'] != rid:
                raise ValueError('Changed query ordering')
            validate_past_join(history, a['image_rows'][global_row], rows[global_row], expected)
            transform = causal_coordinate_transform(history[:,2:],history[:,0],int(index['horizon_raw']))
            if transform['scale'] != a['scale'][global_row]:
                raise ValueError('Past scale differs from original input')
            if available:
                if np.any(frames[begin:current+1] > frames[current]):
                    raise ValueError('Post-query frame requested')
                for slot, i in enumerate(range(begin,current)):
                    if i not in pair_cache:
                        small_a = np.repeat(np.repeat(a['rgb'][offset+i],3,axis=1),3,axis=2)
                        small_b = np.repeat(np.repeat(a['rgb'][offset+i+1],3,axis=1),3,axis=2)
                        low, lowq, pooled = dense_pair(small_a,small_b,cov[i],cov[i+1],centers[i:i+2],h,cv2)
                        high, highq, _ = dense_pair(rgb[i],rgb[i+1],cov[i],cov[i+1],centers[i:i+2],h,cv2)
                        pair_cache[i] = (np.stack([low,high,np.tile(pooled,(16,1))]), np.stack([lowq,highq]))
                    pair_grid, pair_quality = pair_cache[i]
                    for mode in range(3):
                        motion[global_row,mode,slot] = normalized_grid(pair_grid[mode],transform['rotation'],transform['scale'])
                    quality[global_row,:,slot] = pair_quality
            global_row += 1
            if global_row % 500 == 0:
                heartbeat(state='spatial_flow',recording=rid,rows=global_row,pairs=len(pair_cache))
        if available:
            if not pair_receipt.exists():
                keys = sorted(pair_cache)
                np.savez(pair_file,keys=keys,grid=np.stack([pair_cache[k][0] for k in keys]),quality=np.stack([pair_cache[k][1] for k in keys]))
                json_write(pair_receipt,dict(identity=identity,sha256=file_digest(pair_file)))
            offset += len(points)
        summaries[rid] = dict(rows=len(indices), unique_pairs=len(pair_cache),
            result_source='fresh_run_native_pixels_and_spatial_flow' if available else 'not_run_video_missing_rows_retained',
            quality_mean=quality[first_global:global_row].mean((0,2,3)).tolist())
        if metadata:
            summaries[rid]['native_arrays'] = metadata['arrays']
            summaries[rid]['native_source_rows_replayed_exact'] = len(points)
    if global_row != receipt['rows'] or offset != len(a['rgb']) or not np.isfinite(motion).all():
        raise ValueError('Full cohort/cache alignment failed')
    (output/'inputs').mkdir(exist_ok=True)
    for name,array in [('motion.npy',motion),('quality.npy',quality)]:
        np.save(output/'inputs'/name,array,allow_pickle=False)
    report = dict(identity=identity,rows=len(rows),records=summaries,motion_shape=list(motion.shape),
        result_source='fresh_run_native_decode_and_flow_cached_verified_query_schema',
        arrays={name:file_digest(output/'inputs'/name) for name in ('motion.npy','quality.npy')},
        elapsed_seconds=time.monotonic()-started,private_array_bytes=sum(p.stat().st_size for p in output.rglob('*.npy')),
        future_labels_opened=False,development_calibration_confirmation_opened=False,missing_rows_removed=False,
        lowpass_replay_exact=True,source_coordinate_claim='dataset_local_unverified',
        observation_mode='offline_annotated_not_strict_sensor_as_of',stage5c_executed=False,smc_enabled=False)
    json_write(manifest_path,report)
    json_write(reports/'input_report.json',report)
    heartbeat(state='inputs_complete',rows=len(rows),seconds=report['elapsed_seconds'])


if __name__ == '__main__':
    main()
