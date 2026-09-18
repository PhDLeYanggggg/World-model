"""Independently replay auxiliary geometry and sampled source pixels; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.audit_m3w_sdd_state_support import load_source
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import geometry_features
from src.world_model.m3w_sdd_auxiliary import AuxiliaryGeometry, load_registration, source_entries
from src.world_model.m3w_sdd_past_images import border_padding_suspect, supported_center_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--recording')
    args = parser.parse_args()
    reg = load_registration(ROOT,args.registration)
    sys.path.insert(0,str(ROOT/reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Changed decoder')
    entries = source_entries(ROOT,reg); roster = [e['annotation_key'] for e in entries]
    records = []
    started = time.monotonic()
    for entry in entries:
        key = entry['annotation_key']
        if args.recording and key != args.recording:
            continue
        folder = ROOT/reg['output']/'inputs'/key
        receipt = json.loads((folder/'receipt.json').read_text())
        if receipt['identity']['registration_sha256'] != file_digest(args.registration):
            raise ValueError('Changed cache identity')
        for name,digest in receipt['arrays'].items():
            if file_digest(folder/name) != digest:
                raise ValueError('Changed cache '+key+'/'+name)
        for kind in ('video','annotations'):
            if file_digest(ROOT/entry[kind+'_path']) != entry[kind+'_sha256']:
                raise ValueError('Changed source')
        arrays = {name:np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in
            ('geometry','baseline','target','valid','query_keys','image_rows','crop_keys',
             'image_boxes','rgb','coverage','observed_rgb','geometric')}
        n = len(arrays['geometry'])
        expected = arrays['query_keys'][:,0,None]-np.arange(7,-1,-1)*12
        np.testing.assert_array_equal(arrays['crop_keys'][arrays['image_rows'],0],expected)
        np.testing.assert_array_equal(arrays['crop_keys'][arrays['image_rows'],1],
                                      np.broadcast_to(arrays['query_keys'][:,1,None],(n,8)))
        rows,labels = load_source(ROOT/entry['annotations_path'])
        adapter = AuxiliaryGeometry(rows,labels,key,reg,roster)
        checks = 0
        for item in np.unique([0,n//2,n-1]):
            inputs = adapter.inputs(int(item)); target = adapter.labels_for_loss(int(item))
            np.testing.assert_array_equal(arrays['geometry'][item],geometry_features(inputs))
            np.testing.assert_array_equal(arrays['baseline'][item],inputs['baseline_rollouts'][1])
            np.testing.assert_array_equal(arrays['target'][item],target['future_xy_normalized'])
            np.testing.assert_array_equal(arrays['valid'][item],target['future_label_mask'])
            checks += 1
        selected = set(map(int,receipt['sampled_frame_hashes']))
        frames = patches = 0
        with av.open(str(ROOT/entry['video_path'])) as container:
            container.streams.video[0].codec_context.thread_count = 4
            for frame_id,frame in enumerate(container.decode(video=0)):
                if frame_id in selected:
                    image = frame.to_ndarray(format='rgb24')
                    if hashlib.sha256(image.tobytes()).hexdigest() != receipt['sampled_frame_hashes'][str(frame_id)]:
                        raise ValueError('Independent source pixel replay differs')
                    suspect = border_padding_suspect(image,8)
                    for row in np.flatnonzero(arrays['crop_keys'][:,0] == frame_id):
                        box = arrays['image_boxes'][row]
                        patch = supported_center_patch(image,(box[:2]+box[2:])/2,suspect,96,32)
                        for dst,src in (('rgb','rgb_retained'),('observed_rgb','rgb_observed'),
                                        ('coverage','retained_count'),('geometric','geometric_count')):
                            np.testing.assert_array_equal(arrays[dst][row],patch[src])
                        patches += 1
                    frames += 1
                if frame_id >= max(selected):
                    break
        if frames != len(selected):
            raise ValueError('Missing replay frames')
        record = dict(recording=key,all_query_image_joins=n,geometry_label_replays=checks,
                      independent_frames=frames,exact_crop_replays=patches,
                      array_hashes_checked=len(receipt['arrays']))
        records.append(record); print(json.dumps(record),flush=True)
    if args.recording and len(records) != 1:
        raise ValueError('Unknown pilot recording')
    if not args.recording and len(records) != 40:
        raise ValueError('Incomplete source verification')
    name = 'input_verification_pilot.json' if args.recording else 'input_verification.json'
    json_write(ROOT/reg['reports']/name,dict(
        result_source='fresh_run_independent_source_replay_and_cached_hash_verification',
        registration_sha256=file_digest(args.registration),records=records,
        all_queries=sum(r['all_query_image_joins'] for r in records),
        frame_replays=sum(r['independent_frames'] for r in records),
        crop_replays=sum(r['exact_crop_replays'] for r in records),
        geometry_label_replays=sum(r['geometry_label_replays'] for r in records),
        elapsed_seconds=time.monotonic()-started,new_optimizer_updates=0,
        original_val_test_opened=False))


if __name__ == '__main__':
    main()

