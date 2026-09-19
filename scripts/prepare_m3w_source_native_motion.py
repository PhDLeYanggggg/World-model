"""Recover only admitted past native crops, checking exact old-pixel reduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing training libraries')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import load_config, context, FeatureCorpus
from scripts.run_m3w_source_crossfit import immutable_json, array_hash
from scripts.prepare_m3w_sdd_auxiliary import save_array
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_temporal_information import verify_history_keys
from src.world_model.m3w_sdd_auxiliary import load_registration, source_entries
from src.world_model.m3w_sdd_past_images import border_padding_suspect, supported_center_patch
from src.world_model.m3w_spatial_motion import reduce_patch
from src.world_model.m3w_source_motion_resolution import registration, REGISTRATION
from src.world_model.m3w_offline_visual_data import json_write
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-after-records', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    plan = registration(ROOT)
    if shutil.disk_usage(ROOT).free < 4*1024**3:
        raise OSError('Less than 4GiB free; stop without deleting source data')
    aux = load_registration(ROOT, Path('configs/m3w_sdd_auxiliary_v1.json'))
    entries = {e['annotation_key']: e for e in source_entries(ROOT, aux)}
    sys.path.insert(0, str(ROOT/aux['decoder_path'])); import av
    if av.__version__ != aux['decoder_version']: raise ValueError('Decoder changed')
    reg = load_config(Path(plan['source_registration'])); data = context(reg)
    cached = FeatureCorpus(data, ROOT/reg['output'], json.loads((ROOT/reg['reports']/'preparation.json').read_text()))
    ids = cached.ids; loc = ids-data.nmain
    manifest = json.loads(data.manifest_path.read_text())
    private, public = ROOT/plan['output'], ROOT/plan['reports']
    identity = dict(registration_sha256=file_digest(ROOT/REGISTRATION), source=data.identity,
                    assignment=data.assignment_hash, ids_sha256=array_hash(ids), av=av.__version__, numpy=np.__version__)
    immutable_json(private/'decode_identity.json', identity)
    def beat(**value):
        event = dict(pid=os.getpid(), time=time.time(), **value)
        json_write(private/'decode_heartbeat.json', event); print(json.dumps(event), flush=True)
    receipts = []; fresh = 0
    for rid in np.unique(data.record_ids[data.sid[loc]]):
        at = np.flatnonzero(data.record_ids[data.sid[loc]] == rid)
        item = manifest['records'][int(rid)]; key = item['recording']; entry = entries[key]
        folder = data.manifest_path.parent/key
        arrays = {}
        for name in ('query_keys','crop_keys','image_rows','image_boxes','rgb','coverage'):
            path = folder/(name+'.npy')
            if file_digest(path) != item['arrays'][name+'.npy']: raise ValueError('Changed source '+str(path))
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        local = data.local_ids[data.sid[loc[at]]]
        original = arrays['image_rows'][local]
        verify_history_keys(arrays['query_keys'][local], arrays['crop_keys'], original)
        unique = np.unique(original); keys = arrays['crop_keys'][unique]; boxes = arrays['image_boxes'][unique]
        part = private/'native'/f'record{rid}'; part.mkdir(parents=True, exist_ok=True)
        rp = part/'receipt.json'
        part_identity = dict(identity, record=int(rid), crop_indices_sha256=array_hash(unique),
                             video_sha256=entry['video_sha256'], annotations_sha256=entry['annotations_sha256'])
        for kind in ('annotations','video'):
            if file_digest(ROOT/entry[kind+'_path']) != entry[kind+'_sha256']: raise ValueError('Changed raw '+kind)
        if rp.exists():
            receipt = json.loads(rp.read_text()); assert receipt['identity'] == part_identity
            for name, digest in receipt['arrays'].items():
                assert file_digest(part/name) == digest
            receipts.append(receipt); beat(state='record_cached_verified', recording=key); continue
        started = time.monotonic(); n = len(unique)
        rgb = np.zeros((n,3,96,96), np.uint8); cov = np.zeros((n,96,96), np.uint8)
        frames, first, counts = np.unique(keys[:,0], return_index=True, return_counts=True)
        requests = {int(f):(int(a),int(a+c)) for f,a,c in zip(frames,first,counts)}
        if np.any(np.diff(keys[:,0]) < 0): raise ValueError('Expected sorted crop keys')
        limit = int(frames[-1])+1; found = 0; decoded = 0; exact = 0; samples = {}
        old_receipt = json.loads((folder/'receipt.json').read_text())
        with av.open(str(ROOT/entry['video_path'])) as container:
            container.streams.video[0].codec_context.thread_count = 4
            for frame_id, frame in enumerate(container.decode(video=0)):
                decoded += 1
                if frame_id in requests:
                    image = frame.to_ndarray(format='rgb24')
                    if image.shape[:2] != tuple(entry['geometry']['video_size'][::-1]):
                        raise ValueError('Changed source dimensions')
                    suspect = border_padding_suspect(image, 8)
                    begin, end = requests[frame_id]
                    for i in range(begin,end):
                        patch = supported_center_patch(image, (boxes[i,:2]+boxes[i,2:])/2, suspect, 96,96)
                        rgb[i],cov[i] = patch['rgb_retained'],patch['retained_count']
                        low,coverage = reduce_patch(rgb[i],cov[i])
                        np.testing.assert_array_equal(low,arrays['rgb'][unique[i]])
                        np.testing.assert_array_equal(coverage,arrays['coverage'][unique[i]])
                        exact += 1
                    if str(frame_id) in old_receipt['sampled_frame_hashes']:
                        digest = hashlib.sha256(image.tobytes()).hexdigest()
                        assert digest == old_receipt['sampled_frame_hashes'][str(frame_id)]
                        samples[str(frame_id)] = digest
                    found += 1
                if frame_id % 2000 == 0:
                    beat(state='decoding',recording=key,frame=frame_id,last_requested=limit-1,exact_crops=exact)
                if frame_id == limit-1: break
        if decoded != limit or found != len(requests) or exact != n:
            raise ValueError('Missing source frames/crops; no silent removal')
        hashes = dict(save_array(part,name,value) for name,value in
                      dict(rgb=rgb,coverage=cov,original_rows=unique).items())
        receipt = dict(identity=part_identity,record=int(rid),recording=key,arrays=hashes,
            crops=n,queries=len(at),decoded_frames=decoded,requested_frames=found,exact_reductions=exact,
            sampled_frame_hashes=samples,seconds=time.monotonic()-started,
            bytes=sum((part/name).stat().st_size for name in hashes),future_images_used=False)
        immutable_json(rp,receipt); receipts.append(receipt); fresh += 1
        beat(state='record_complete',recording=key,seconds=receipt['seconds'],crops=n)
        if args.stop_after_records and fresh >= args.stop_after_records:
            beat(state='pilot_complete_no_labels_or_fitting',new_records=fresh); return
    if len(receipts) != 29 or sum(r['crops'] for r in receipts) != 25300:
        raise ValueError('Registered crop population changed')
    immutable_json(public/'preparation.json',dict(identity=identity,result_source='fresh_run_native_decode',
        records=receipts,rows=len(ids),crops=25300,exact_reductions=25300,
        bytes=sum(r['bytes'] for r in receipts),seconds=sum(r['seconds'] for r in receipts),
        future_labels_used=False,main_outer_rows_scored=0,new_deployment=False))
    beat(state='complete',new_records=fresh,crops=25300)


if __name__ == '__main__': main()
