"""Sampled native-frame replay, fixed pair replay and input-support diagnosis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_m3w_observed_motion import load_registration
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_masked_history_images import masked_center_patch
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_spatial_motion import dense_pair, reduce_patch, feature_variant
from src.world_model.m3w_objective_alignment import supported_standardization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--decode-only', action='store_true')
    args = p.parse_args()
    reg, parent, contract, source, _ = load_registration(args.registration)
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    receipt = json.loads((output/'inputs/manifest.json').read_text())
    if args.decode_only:
        sys.path.insert(0, str(ROOT/parent['decoder_path']))
        import av
        checked = {}
        for rid in parent['recordings']:
            if rid == 'ucy_zara03':
                continue
            reader, _ = contract.open_recording(rid,purpose='fit')
            directory = (ROOT/reader.metadata['files'][0]['path']).parent
            native = output/'native'/rid
            rgb = np.load(native/'rgb.npy',mmap_mode='r')
            cov = np.load(native/'coverage.npy',mmap_mode='r')
            centers, frames = np.load(native/'image_xy.npy'), np.load(native/'source_frame.npy')
            selected = np.unique(np.linspace(0,len(frames)-1,8,dtype=int))
            requests = {}
            for i in selected:
                requests.setdefault(int(frames[i]),[]).append(i)
            matched = 0
            with av.open(str(directory/'video.avi')) as video:
                for j, frame in enumerate(video.decode(video=0)):
                    if j in requests:
                        image = np.asarray(frame.to_image())
                        for i in requests[j]:
                            actual, mask = masked_center_patch(image,centers[i],96,96)
                            if not np.array_equal(actual,rgb[i]) or not np.array_equal(mask,cov[i]):
                                raise ValueError('Independent native frame replay differs')
                            matched += 1
                    if j >= max(requests):
                        break
            if matched != len(selected):
                raise ValueError('Sampled raw source incomplete')
            checked[rid] = dict(sampled_native_rows=matched,rgb_and_mask_exact=True)
        json_write(reports/'native_decode_replay.json',dict(result_source='fresh_run_independent_decode',records=checked,
            cv2_loaded=False,sample_only_not_full_second_decode=True))
        return
    completed = subprocess.run([sys.executable,__file__,'--registration',str(args.registration),'--decode-only'],cwd=ROOT)
    if completed.returncode:
        raise RuntimeError('Independent decoder failed')
    sys.path.insert(0,str(ROOT/reg['opencv_runtime']))
    import cv2
    cv2.setNumThreads(1)
    checks = {}
    for rid in parent['recordings']:
        if rid == 'ucy_zara03':
            continue
        reader,_ = contract.open_recording(rid,purpose='fit')
        h = np.loadtxt((ROOT/reader.metadata['files'][0]['path']).parent/'H.txt')
        native = output/'native'/rid
        rgb,cov = np.load(native/'rgb.npy',mmap_mode='r'),np.load(native/'coverage.npy',mmap_mode='r')
        centers = np.load(native/'image_xy.npy')
        with np.load(output/(rid+'_pairs.npz')) as pairs:
            positions = np.unique(np.linspace(0,len(pairs['keys'])-1,16,dtype=int))
            for slot in positions:
                i = int(pairs['keys'][slot])
                high,hq,_ = dense_pair(rgb[i],rgb[i+1],cov[i],cov[i+1],centers[i:i+2],h,cv2)
                x,_ = reduce_patch(rgb[i],cov[i])
                y,_ = reduce_patch(rgb[i+1],cov[i+1])
                low,lq,pooled = dense_pair(x.repeat(3,1).repeat(3,2),y.repeat(3,1).repeat(3,2),
                                           cov[i],cov[i+1],centers[i:i+2],h,cv2)
                if not np.array_equal(np.stack([low,high,np.tile(pooled,(16,1))]),pairs['grid'][slot]):
                    raise ValueError('Fixed observed flow replay differs')
                if not np.array_equal(np.stack([lq,hq]),pairs['quality'][slot]):
                    raise ValueError('Fixed flow support differs')
            checks[rid] = dict(pairs_replayed=len(positions),exact=True)
    geom = np.load(source/'geometry.npy',mmap_mode='r')
    folds = np.load(source/'folds.npy')
    motion,quality = np.load(output/'inputs/motion.npy'),np.load(output/'inputs/quality.npy')
    static = np.all(geom[:,:16] == 0,axis=1)
    support = {}
    for variant in reg['variants']:
        x = feature_variant(geom,motion,quality,variant)
        support[variant] = {}
        for fold in range(3):
            train,held = np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            _, fit = supported_standardization(x[train],x[held])
            z = (x[held]-fit['mean'])/fit['std']
            z[:,fit['constant']] = 0
            clipped = (np.abs(z[:,-336:])>10).any(1)
            support[variant][fold] = dict(any_motion_column_clipped_fraction=float(clipped.mean()),
                stationary_motion_clipped_fraction=float(clipped[static[held]].mean()) if static[held].any() else None)
    difference = np.linalg.norm(motion[:,1,:,:,:2]-motion[:,0,:,:,:2],axis=-1)
    result = dict(result_source='fresh_run_fixed_input_replay_and_support_diagnostic',
        input_manifest_sha256=file_digest(output/'inputs/manifest.json'),sampled_flow_pairs=checks,
        support=support,native_lowpass_vector_difference_quantiles_normalized=np.quantile(difference,[0,.5,.9,.99,1]).tolist(),
        future_labels_opened=False,source_role='fit_only',new_model_or_threshold_selection=False,
        no_extra_scene_support_added=True,initial_build_warning='OpenCV/PyAV duplicate AVFoundation class warning; sampled decode replay uses a separate no-OpenCV process',
        development_calibration_confirmation_opened=False)
    json_write(reports/'input_diagnosis.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
