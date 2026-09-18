"""Build full approved SDD inputs, with separate masked targets."""
from __future__ import annotations
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
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.audit_m3w_sdd_state_support import load_source
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import geometry_features
from src.world_model.m3w_sdd_auxiliary import AuxiliaryGeometry, load_registration, source_entries
from src.world_model.m3w_sdd_image_coordinates import SDDImageCoordinates
from src.world_model.m3w_sdd_past_images import border_padding_suspect, supported_center_patch


def save_array(folder, name, value):
    path = folder/(name+'.npy'); temporary = folder/(name+'.tmp.npy')
    np.save(temporary, value, allow_pickle=False); os.replace(temporary, path)
    return path.name, file_digest(path)


def prepare_record(entry, reg, identity, roster, av, heartbeat):
    key = entry['annotation_key']; folder = ROOT/reg['output']/'inputs'/key
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder/'receipt.json'
    for kind in ('annotations', 'video'):
        if file_digest(ROOT/entry[kind+'_path']) != entry[kind+'_sha256']:
            raise ValueError('Changed auxiliary source '+kind+': '+key)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['identity'] != identity:
            raise ValueError('Changed auxiliary preparation identity')
        for name, digest in receipt['arrays'].items():
            if file_digest(folder/name) != digest:
                raise ValueError('Changed auxiliary array '+key+'/'+name)
        return receipt, False
    started = time.monotonic()
    heartbeat(dict(state='source_read', recording=key))
    rows, labels = load_source(ROOT/entry['annotations_path'])
    source = AuxiliaryGeometry(rows, labels, key, reg, roster)
    adapter = source.adapter
    np.testing.assert_array_equal(adapter.index, np.load(
        ROOT/reg['geometry_bridge']/key/'index_stride12.npy', allow_pickle=False))
    n = len(adapter)
    values = dict(geometry=np.empty((n, 476), np.float32),
        baseline=np.empty((n, 12, 2), np.float32), target=np.zeros((n, 12, 2), np.float32),
        valid=np.zeros((n, 12), bool), scale=np.empty(n, np.float64))
    history_points = adapter.index['current_row'][:, None]-np.arange(7, -1, -1)
    past_source_ids = adapter.source_ids[history_points]
    unique_source = np.unique(past_source_ids)
    order = np.lexsort((adapter.source[unique_source, 0], adapter.source[unique_source, 5]))
    chosen_ids = unique_source[order]; chosen = adapter.source[chosen_ids]
    source_lookup = np.full(len(adapter.source), -1, np.int64)
    source_lookup[chosen_ids] = np.arange(len(chosen))
    values['image_rows'] = source_lookup[past_source_ids]
    current = adapter.points[adapter.index['current_row']]
    values['query_keys'] = current[:, :2].astype(np.int64)
    expected_frames = current[:, 0, None]-np.arange(7, -1, -1)*12
    np.testing.assert_array_equal(chosen[values['image_rows'], 5], expected_frames)
    np.testing.assert_array_equal(chosen[values['image_rows'], 0], np.broadcast_to(current[:, 1, None], (n, 8)))
    for item in range(n):
        inputs = source.inputs(item)
        if np.any(inputs['history_frame_offsets'] > 0) or np.any(inputs['neighbor_frame_offsets'][inputs['neighbor_mask']] > 0):
            raise ValueError('Future observation in auxiliary inputs')
        values['geometry'][item] = geometry_features(inputs)
        values['baseline'][item] = inputs['baseline_rollouts'][1]
        values['scale'][item] = inputs['causal_features'][2]
        # Open loss targets only after constructing the complete inference payload.
        target = source.labels_for_loss(item)
        values['target'][item], values['valid'][item] = target['future_xy_normalized'], target['future_label_mask']
        if item % 2000 == 0:
            heartbeat(dict(state='geometry_labels_separate', recording=key, query=item, total=n))
    if not all(np.isfinite(values[k]).all() for k in ('geometry', 'baseline', 'target', 'scale')):
        raise ValueError('Nonfinite auxiliary data')
    mapper = SDDImageCoordinates(*entry['geometry']['annotation_size'], *entry['geometry']['video_size'])
    limit = int(chosen[:, 5].max())+1
    boxes = mapper.past_boxes(chosen[:, 1:5], frame_ids=chosen[:, 5], query_frame=limit-1)
    crop_arrays = dict(rgb=np.zeros((len(chosen), 3, 32, 32), np.uint8),
        observed_rgb=np.zeros((len(chosen), 3, 32, 32), np.uint8),
        coverage=np.zeros((len(chosen), 32, 32), np.uint8),
        geometric=np.zeros((len(chosen), 32, 32), np.uint8))
    unique_frames, first, counts = np.unique(chosen[:, 5].astype(int), return_index=True, return_counts=True)
    requests = {int(f): (int(a), int(a+c)) for f,a,c in zip(unique_frames, first, counts)}
    decoded, found, samples = 0, 0, {}
    sample_frames = {unique_frames[i] for i in np.unique([0, len(unique_frames)//2, len(unique_frames)-1])}
    with av.open(str(ROOT/entry['video_path'])) as container:
        container.streams.video[0].codec_context.thread_count = 4
        for frame_id, frame in enumerate(container.decode(video=0)):
            decoded += 1
            if frame_id in requests:
                image = frame.to_ndarray(format='rgb24')
                if image.shape[:2] != (mapper.video_height, mapper.video_width):
                    raise ValueError('Source video dimensions changed')
                suspect = border_padding_suspect(image, 8)
                begin, end = requests[frame_id]
                for i in range(begin, end):
                    patch = supported_center_patch(image, (boxes[i, :2]+boxes[i, 2:])/2, suspect, 96, 32)
                    for dst, src in (('rgb','rgb_retained'), ('observed_rgb','rgb_observed'),
                                     ('coverage','retained_count'), ('geometric','geometric_count')):
                        crop_arrays[dst][i] = patch[src]
                if frame_id in sample_frames:
                    samples[str(frame_id)] = hashlib.sha256(image.tobytes()).hexdigest()
                found += 1
            if frame_id % 2000 == 0:
                heartbeat(dict(state='past_pixel_extraction', recording=key, frame=frame_id, last_requested=limit-1))
            if frame_id == limit-1:
                break
    if decoded != limit or found != len(requests):
        raise ValueError('Missing video frames; no silent row removal')
    values.update(crop_arrays)
    values['crop_keys'] = chosen[:, [5, 0]].astype(np.int64)
    values['source_flags'] = chosen[:, 6:9].astype(np.uint8)
    values['annotation_boxes'], values['image_boxes'] = chosen[:, 1:5], boxes
    checks = 0
    for item in np.unique([0, n//2, n-1]):
        query = adapter.identity(int(item)); cutoff = query['frame_id']
        changed = rows.copy(); future = changed[:, 5] > cutoff
        changed[future, 1:5] += 12345; changed[future, 6:9] = 1
        other = AuxiliaryGeometry(changed, labels, key, reg, roster).adapter
        points = other.points[other.index['current_row']]
        match = np.flatnonzero((points[:, 0] == cutoff) & (points[:, 1] == query['agent_id']))
        if len(match) != 1:
            raise ValueError('Future corruption changed past query population')
        for name, value in adapter.get_inputs(int(item)).items():
            np.testing.assert_array_equal(value, other.get_inputs(int(match[0]))[name])
        checks += 1
    hashes = dict(save_array(folder, name, value) for name,value in values.items())
    valid = values['valid']; geometric = values['geometric'].sum((1,2)); kept = values['coverage'].sum((1,2))
    receipt = dict(identity=identity, recording=key, source_rows=len(rows),
        rows=n, crops=len(chosen), data_role='supervised_auxiliary_training', original_split='train',
        annotation_sha256=entry['annotations_sha256'], video_sha256=entry['video_sha256'],
        arrays=hashes, complete_labels=int(valid.all(1).sum()),
        partial_labels=int((valid.any(1) & ~valid.all(1)).sum()), absent_labels=int((~valid.any(1)).sum()),
        geometrically_partial_crops=int((geometric < 96**2).sum()),
        suspect_border_crops=int((kept < geometric).sum()), no_pixel_crops=int((kept == 0).sum()),
        decoded_frames=decoded, requested_frames=found, sampled_frame_hashes=samples,
        future_mutation_checks=checks, elapsed_seconds=time.monotonic()-started,
        bytes=sum((folder/name).stat().st_size for name in hashes), optimizer_updates=0)
    json_write(receipt_path, receipt)
    return receipt, True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True); p.add_argument('--recording')
    args = p.parse_args()
    reg = load_registration(ROOT, args.registration)
    entries = source_entries(ROOT, reg); roster = [e['annotation_key'] for e in entries]
    if args.recording and args.recording not in roster:
        raise ValueError('Recording is not in original train-40')
    if shutil.disk_usage(ROOT).free < 8*1024**3:
        raise OSError('Less than 8 GiB free; preserve source data and stop')
    sys.path.insert(0, str(ROOT/reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Decoder version changed')
    identity = dict(registration_sha256=file_digest(args.registration), numpy=np.__version__, av=av.__version__)
    output = ROOT/reg['output']
    def heartbeat(value):
        json_write(output/'prepare_heartbeat.json', dict(pid=os.getpid(), updated_unix=time.time(), **value))
        print(json.dumps(value), flush=True)
    records, fresh = [], 0
    for entry in entries:
        if args.recording and entry['annotation_key'] != args.recording:
            continue
        receipt, new = prepare_record(entry, reg, identity, roster, av, heartbeat)
        records.append(receipt); fresh += int(new)
        heartbeat(dict(state='record_complete', recording=entry['annotation_key'], rows=receipt['rows'],
                       crops=receipt['crops'], seconds=receipt['elapsed_seconds'], fresh=new))
    if args.recording:
        return
    if sum(r['rows'] for r in records) != 229333:
        raise ValueError('Full stride-12 cohort changed')
    manifest = dict(identity=identity, records=records, rows=sum(r['rows'] for r in records),
        crops=sum(r['crops'] for r in records), source_role='supervised_auxiliary_training',
        stride=12, observed=8, predicted=12, original_val_test_opened=False,
        complete_labels=sum(r['complete_labels'] for r in records),
        partial_labels=sum(r['partial_labels'] for r in records),
        absent_labels=sum(r['absent_labels'] for r in records),
        bytes=sum(r['bytes'] for r in records), new_model_training=False)
    path = output/'inputs/manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Completed auxiliary manifest changed')
    else:
        json_write(path, manifest)
    public = {k:v for k,v in manifest.items() if k != 'records'}
    public['records'] = [{k:v for k,v in r.items() if k not in ('sampled_frame_hashes','arrays')} for r in records]
    public['manifest_sha256'] = file_digest(path)
    json_write(ROOT/reg['reports']/'data_receipt.json', public)
    heartbeat(dict(state='complete', recordings=40, rows=manifest['rows'], crops=manifest['crops'],
                   fresh=fresh, reused=40-fresh, bytes=manifest['bytes']))


if __name__ == '__main__':
    main()

