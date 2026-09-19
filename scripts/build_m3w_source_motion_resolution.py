"""Extract four fixed motion measurements from hash-verified observed crops."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import load_config, context, FeatureCorpus
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays, array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_temporal_information import verify_history_keys, describe
from src.world_model.m3w_source_motion_resolution import registration, REGISTRATION, VARIANTS, pair_features
from src.world_model.m3w_offline_visual_data import json_write
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--replay',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    plan=registration(ROOT); private,public=ROOT/plan['output'],ROOT/plan['reports']
    runtime=ROOT/'data/stage_cvpr2027_experiments/optical_flow_runtime'
    if file_digest(runtime/'cv2/cv2.abi3.so') != plan['opencv_sha256']: raise ValueError('Changed OpenCV')
    sys.path.insert(0,str(runtime)); import cv2
    if cv2.__version__ != '4.13.0': raise ValueError('Changed OpenCV version')
    cv2.setNumThreads(1)
    reg=load_config(Path(plan['source_registration'])); data=context(reg)
    cached=FeatureCorpus(data,ROOT/reg['output'],json.loads((ROOT/reg['reports']/'preparation.json').read_text()))
    ids=cached.ids; loc=ids-data.nmain; manifest=json.loads(data.manifest_path.read_text())
    prep=json.loads((public/'preparation.json').read_text())
    links={v['annotation_key']:v for v in json.loads((ROOT/plan['media_links']).read_text())['records']}
    old=json.loads((ROOT/plan['old_flow_report']).read_text())
    for r in old['records']:
        if file_digest(ROOT/r['path']) != r['sha256']: raise ValueError('Changed old flow control')
    identity=dict(registration_sha256=file_digest(ROOT/REGISTRATION),source=data.identity,
        assignment=data.assignment_hash,ids_sha256=array_hash(ids),preparation_sha256=file_digest(public/'preparation.json'))
    output={v:np.zeros((len(ids),7,19),np.float32) for v in VARIANTS}; receipts=[]; replayed=0; new=0
    for record in prep['records']:
        rid=record['record']; at=np.flatnonzero(data.record_ids[data.sid[loc]]==rid)
        item=manifest['records'][rid]; folder=data.manifest_path.parent/item['recording']
        a={}
        for name in ('query_keys','crop_keys','image_rows','image_boxes','source_flags','rgb','coverage'):
            path=folder/(name+'.npy'); assert file_digest(path)==item['arrays'][name+'.npy']
            a[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        rows=a['image_rows'][data.local_ids[data.sid[loc[at]]]]
        verify_history_keys(a['query_keys'][data.local_ids[data.sid[loc[at]]]],a['crop_keys'],rows)
        pairs,inverse=np.unique(np.stack((rows[:,:-1],rows[:,1:]),-1).reshape(-1,2),axis=0,return_inverse=True)
        native_folder=private/'native'/f'record{rid}'
        for name,digest in record['arrays'].items(): assert file_digest(native_folder/name)==digest
        native={n:np.load(native_folder/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('rgb','coverage','original_rows')}
        mapped=np.searchsorted(native['original_rows'],pairs)
        np.testing.assert_array_equal(native['original_rows'][mapped],pairs)
        geometry=links[item['recording']]['geometry']
        scale=np.asarray(geometry['video_size'])/np.asarray(geometry['annotation_size'])
        previous=next(r for r in old['records'] if r['identity']['record']==rid)
        with np.load(ROOT/previous['path'],allow_pickle=False) as prior:
            np.testing.assert_array_equal(prior['pairs'],pairs); control=prior['features'].copy()
        for variant in VARIANTS:
            path=private/'pairs'/f'{rid}_{variant}.npz'; rp=path.with_suffix('.json')
            ti=dict(identity,record=rid,variant=variant,pairs_sha256=array_hash(pairs),scale_xy=scale.tolist())
            receipt=json.loads(rp.read_text()) if rp.exists() else None
            if receipt:
                assert receipt['identity']==ti and file_digest(path)==receipt['sha256']
                with np.load(path,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['pairs'],pairs); values=saved['features'].copy()
            if receipt is None or args.replay:
                if args.replay and receipt is None: raise ValueError('Incomplete flow cannot be replayed')
                start=time.monotonic(); measured=np.empty((len(pairs),19),np.float32)
                for i,pair in enumerate(pairs):
                    source=native if variant.startswith('native') else a
                    ix=mapped[i] if variant.startswith('native') else pair
                    measured[i]=pair_features(source['rgb'][ix],source['coverage'][ix],a['image_boxes'][pair],
                        a['source_flags'][pair],scale,cv2,variant)
                if variant=='lowpass_w45': np.testing.assert_array_equal(measured,control)
                if args.replay:
                    np.testing.assert_array_equal(measured,values); replayed+=len(pairs)
                else:
                    values=measured; save_arrays(path,dict(pairs=pairs,features=values))
                    receipt=dict(identity=ti,pairs=len(pairs),seconds=time.monotonic()-start,
                                 path=str(path.relative_to(ROOT)),sha256=file_digest(path))
                    immutable_json(rp,receipt); new+=len(pairs)
            output[variant][at]=values[inverse].reshape(-1,7,19); receipts.append(receipt)
            event=dict(pid=os.getpid(),time=time.time(),recording=item['recording'],variant=variant,
                       state='pair_record_complete',new_pairs=new,replayed_pairs=replayed)
            json_write(private/'flow_heartbeat.json',event); print(json.dumps(event),flush=True)
    if args.replay:
        immutable_json(public/'flow_replay.json',dict(identity=identity,exact_pair_replays=replayed,new_extractions=0)); return
    artifacts={}
    for variant,values in output.items():
        if not np.isfinite(values).all(): raise ValueError('Nonfinite flow')
        path=private/(variant+'.npz'); save_arrays(path,dict(ids=ids,features=values))
        artifacts[variant]=dict(path=str(path.relative_to(ROOT)),sha256=file_digest(path),
            box_support=float(values[...,14].mean()),surround_support=float(values[...,15].mean()),
            box_magnitude=describe(values[...,6].ravel()),surround_magnitude=describe(values[...,8].ravel()))
    immutable_json(public/'extraction.json',dict(identity=identity,result_source='fresh_run_observed_flow',
        rows=len(ids),unique_pairs_per_variant=23890,variants=artifacts,records=receipts,
        exact_old_control_pairs=23890,future_labels_used=False,main_outer_rows_scored=0))
    print(json.dumps(dict(state='complete',new_pair_measurements=new)))


if __name__=='__main__': main()
