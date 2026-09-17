"""Extract only the eight observed RGB patches for frozen stationary fit rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_past_video_alignment import native_to_image_xy, past_frame_indices
from src.evaluation.m3w_past_motion_correspondence import centered_patch
from scripts.run_m3w_stationary_start_probe import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    for p,h in reg['bindings'].items():
        if file_digest(ROOT/p) != h:
            raise ValueError('Changed source: '+p)
    parent = ExperimentContract(json.loads((ROOT/reg['parent_protocol']).read_text()),ROOT)
    if parent.digest != reg['parent_protocol_sha256']:
        raise ValueError('Parent changed')
    source = ROOT/reg['source_cache']
    if file_digest(source) != reg['source_cache_sha256']:
        raise ValueError('Frozen stationary rows changed')
    with np.load(source,allow_pickle=False) as a:
        rows = json.loads(str(a['rows_json']))
        features, basis, scene_scale = [a[k].copy() for k in ('features','basis','scale')]
    # Label-bearing source metadata is reduced to this allowlist before extraction.
    fields = ('recording_id','physical_scene','agent_id','frame_id','fit_fold','native_frame_step','data_role')
    meta = [{k:r[k] for k in fields} for r in rows]
    if any(r['data_role'] != 'fit' or parent.protocol['assignments'][r['recording_id']] != 'fit' for r in meta):
        raise ValueError('Only fit rows permitted')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments') or output.exists():
        raise ValueError('Use new ignored local cache directory')
    output.mkdir(parents=True)
    started = time.monotonic()
    sys.path.insert(0,str(ROOT/reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Decoder changed')
    n = len(meta)
    rgb = np.zeros((n,8,3,32,32),dtype=np.uint8)
    mask = np.zeros((n,8),dtype=bool)
    frame_ids = np.zeros((n,8),dtype=np.int64)
    image_xy, homography, jacobian = np.zeros((n,2)), np.zeros((n,3,3)), np.zeros((n,4))
    scene_reports = {}
    for rid in sorted({r['recording_id'] for r in meta}):
        reader,_ = parent.open_recording(rid,purpose='fit')
        directory = (ROOT/reader.metadata['files'][0]['path']).parent
        h = np.loadtxt(directory/'H.txt')
        source_lookup = {(int(p[0]),int(p[1])):p[2:4] for p in reader.points}
        requests = {}
        indices = [i for i,r in enumerate(meta) if r['recording_id']==rid]
        for i in indices:
            r = meta[i]
            times = past_frame_indices(r['frame_id'],r['frame_id']+np.arange(-7,1)*r['native_frame_step'])
            native = np.array([source_lookup[(int(t),r['agent_id'])] for t in times])
            if not np.array_equal(native,np.broadcast_to(native[-1],native.shape)):
                raise ValueError('Frozen stationary cohort changed')
            pixel = native_to_image_xy(native,h,projected_axes='row_col')
            frame_ids[i], image_xy[i], homography[i] = times,pixel[-1],h
            # Local image-to-scene Jacobian supplies camera axes, never future motion.
            local = pixel[-1]+np.array([[0.,0.],[1.,0.],[0.,1.]])
            projected = np.c_[local[:,::-1],np.ones(3)]@h.T
            native_local = projected[:,:2]/projected[:,2:]
            jacobian[i] = ((native_local[1:]-native_local[0])@basis[i]/scene_scale[i]).reshape(4)
            for j,t in enumerate(times):
                requests.setdefault(int(t),[]).append((i,j,pixel[j]))
        found = set()
        with av.open(str(directory/'video.avi')) as video:
            for index,frame in enumerate(video.decode(video=0)):
                if index in requests:
                    found.add(index)
                    image = np.asarray(frame.to_image())
                    for i,j,p in requests[index]:
                        patch = centered_patch(image,p,96)
                        if patch is not None:
                            small = Image.fromarray(patch).resize((32,32),Image.Resampling.BILINEAR)
                            rgb[i,j] = np.asarray(small).transpose(2,0,1)
                            mask[i,j] = True
                if index % 1000 == 0:
                    atomic_json(output/'heartbeat.json',{'state':'extracting_past_only','pid':os.getpid(),
                        'recording':rid,'decoded_frame':index,'elapsed_seconds':time.monotonic()-started})
                if index >= max(requests):
                    break
        if found != set(requests):
            raise ValueError('Missing requested video indices')
        scene_reports[rid] = {'rows':len(indices),'requested_images':len(indices)*8,
            'unique_frames':len(requests),'valid_patches':int(mask[indices].sum()),
            'rows_full_support':int(mask[indices].all(1).sum()),
            'video_sha256':file_digest(directory/'video.avi')}
    cache = output/'past_inputs.npz'
    np.savez(cache,rgb=rgb,mask=mask,frame_ids=frame_ids,geometry=np.c_[features,jacobian],
             image_xy=image_xy,homography=homography,rows_json=np.array(json.dumps(meta)))
    report = {'result_source':'fresh_run_fit_only_past_RGB_extraction',
        'registration_sha256':file_digest(args.registration),'parent_protocol_sha256':parent.digest,
        'source_cache_sha256':file_digest(source),'cache_sha256':file_digest(cache),
        'rows':n,'records':scene_reports,'future_fields_in_input_metadata':False,
        'future_targets_extracted':False,'past_input_fields':['rgb','mask','frame_ids','geometry','image_xy','homography'],
        'annotation_clock_status':'native_index_assumption_not_physical_capture_clock_verification',
        'source_media_in_git':False,'elapsed_seconds':time.monotonic()-started}
    atomic_json(output/'receipt.json',report)
    atomic_json(output/'heartbeat.json',{'state':'complete','pid':os.getpid(),'elapsed_seconds':time.monotonic()-started})
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
