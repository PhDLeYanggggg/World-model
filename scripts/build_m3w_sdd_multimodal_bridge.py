"""Extract registered past-image requests across SDD train videos, without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 Python')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.audit_m3w_sdd_state_support import load_source
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_sdd_image_coordinates import SDDImageCoordinates
from src.world_model.m3w_sdd_multimodal_bridge import SDDStepImageStore
from src.world_model.m3w_sdd_past_images import ARRAYS, border_padding_suspect, supported_center_patch
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter


def heartbeat(root, **values):
    json_write(root/'heartbeat.json', dict(pid=os.getpid(), updated_unix=time.time(), **values))


def requests(rows, labels, entry, config, bridge):
    adapters, queries, source_ids = {}, [], []
    for stride in config['strides']:
        adapter = SDDStepAdapter(rows, labels, entry['annotation_key'], stride)
        cached = ROOT/bridge['output']/entry['annotation_key']/f'index_stride{stride}.npy'
        np.testing.assert_array_equal(adapter.index, np.load(cached, mmap_mode='r', allow_pickle=False))
        adapters[stride] = adapter
        selected = np.unique(np.linspace(0, len(adapter)-1, min(config['queries_per_stride'], len(adapter)), dtype=int))
        for item in selected:
            query = adapter.identity(int(item))
            index = adapter.index[item]
            ids = adapter.source_ids[int(index['history_start']):int(index['current_row'])+1]
            if np.any(adapter.source[ids, 5] > query['frame_id']):
                raise ValueError('Post-query image request')
            source_ids.extend(ids.tolist())
            queries.append(dict(**query, item=int(item)))
    chosen = rows[np.unique(source_ids)]
    chosen = chosen[np.lexsort((chosen[:, 0], chosen[:, 5]))]
    return adapters, queries, chosen


def check_model_join(store, adapters, queries):
    torch.manual_seed(17)
    model = OfflineVisualForecast(476).eval()
    checked, complete, partial, absent = 0, 0, 0, 0
    for start in range(0, len(queries), 64):
        sample = queries[start:start+64]
        joined = [store.model_inputs(adapters[q['raw_frame_stride']], q['item']) for q in sample]
        geometry, rgb, coverage, baseline = [torch.from_numpy(np.stack([x[key] for x in joined]))
            for key in ('geometry', 'rgb', 'coverage', 'baseline')]
        if not all(torch.isfinite(x).all() for x in (geometry, rgb, coverage, baseline)):
            raise ValueError('Nonfinite multimodal input')
        with torch.no_grad():
            encoded = model.image(torch.cat((rgb, coverage), dim=2).flatten(0, 1))
            if not torch.isfinite(encoded).all():
                raise ValueError('Nonfinite visual interface')
            pred = model(geometry, rgb, coverage, baseline, 'past_rgb')
            torch.testing.assert_close(pred, baseline, rtol=0, atol=0)
        support = coverage.sum((2, 3, 4)) > 0
        complete += int(support.all(1).sum())
        partial += int((support.any(1) & ~support.all(1)).sum())
        absent += int((~support.any(1)).sum())
        checked += len(sample)
    return dict(joined_queries=checked, complete_retained_histories=complete,
                partial_retained_histories=partial, no_retained_histories=absent,
                untrained_visual_forward_exact_CV=True, optimizer_updates=0)


def save_review(directory, store, queries, images):
    selected = [queries[i] for i in np.unique([0, len(queries)//2, len(queries)-1])]
    sheet = Image.new('RGB', (900, 365*len(selected)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, q in enumerate(selected):
        top = i*365
        image = Image.fromarray(images[q['frame_id']])
        mapper = SDDImageCoordinates(*store.metadata['source']['geometry']['annotation_size'],
                                    *store.metadata['source']['geometry']['video_size'])
        history = store.inputs(query_frame=q['frame_id'], agent_id=q['agent_id'], step=q['raw_frame_stride'])
        box = mapper.past_boxes(history['annotation_boxes'][-1:], frame_ids=np.array([q['frame_id']]),
                                query_frame=q['frame_id'])[0]
        ImageDraw.Draw(image).rectangle(tuple(box), outline='red', width=3)
        image.thumbnail((440, 290)); sheet.paste(image, (0, top+25))
        title = f"{store.metadata['source']['annotation_key']} agent {q['agent_id']} query {q['frame_id']} stride {q['raw_frame_stride']}"
        draw.text((3, top+3), title, fill='black')
        for k, frame in enumerate(history['source_frames']):
            x, y = 452+(k % 4)*110, top+30+(k//4)*145
            patch = Image.fromarray(history['rgb_retained'][k].transpose(1, 2, 0))
            sheet.paste(patch.resize((96, 96), Image.Resampling.NEAREST), (x, y))
            draw.text((x, y+100), 'frame '+str(frame), fill='black')
            fraction = history['retained_count'][k].sum()/96**2
            draw.text((x, y+114), f'support {fraction:.2f}', fill='black')
        draw.text((3, top+330), 'Red box: mapped query annotation. Past ego-centered crops; inferred padding mask, not visibility truth.', fill='black')
    sheet.save(directory/'private_contact_sheet.png')


def build_record(entry, config, bridge, identity, av, output):
    key = entry['annotation_key']; directory = output/key
    receipt_path = directory/'receipt.json'
    for name in ('annotations', 'video', 'reference'):
        if file_digest(ROOT/entry[name+'_path']) != entry[name+'_sha256']:
            raise ValueError('Changed source: '+key+'/'+name)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['identity'] != identity:
            raise ValueError('Completed image cache identity changed')
        store = SDDStepImageStore(directory, observation_mode='offline_annotated')
        for name in ('metadata.json', 'queries.json', 'private_contact_sheet.png'):
            if file_digest(directory/name) != receipt['artifacts'][name]:
                raise ValueError('Changed saved image artifact')
        return receipt, False
    start = time.monotonic(); directory.mkdir(parents=True, exist_ok=True)
    rows, labels = load_source(ROOT/entry['annotations_path'])
    adapters, queries, chosen = requests(rows, labels, entry, config, bridge)
    mapper = SDDImageCoordinates(*entry['geometry']['annotation_size'], *entry['geometry']['video_size'])
    limit = int(chosen[:, 5].max())+1
    boxes = mapper.past_boxes(chosen[:, 1:5], frame_ids=chosen[:, 5], query_frame=limit-1)
    n, size = len(chosen), config['output_size']
    arrays = dict(row_keys=chosen[:, [5, 0]].astype(np.int64), annotation_boxes=chosen[:, 1:5],
                  image_boxes=boxes, source_flags=chosen[:, 6:9].astype(np.uint8))
    for name in ('rgb_observed', 'rgb_retained'):
        arrays[name] = np.zeros((n, 3, size, size), np.uint8)
    for name in ('geometric_count', 'retained_count'):
        arrays[name] = np.zeros((n, size, size), np.uint8)
    frame_ids, first, counts = np.unique(chosen[:, 5].astype(int), return_index=True, return_counts=True)
    frame_requests = {int(f):np.arange(a, a+c) for f, a, c in zip(frame_ids, first, counts)}
    review_frames = {queries[i]['frame_id'] for i in np.unique([0, len(queries)//2, len(queries)-1])}
    images, frame_hashes, decoded = {}, {}, 0
    with av.open(str(ROOT/entry['video_path'])) as container:
        container.streams.video[0].codec_context.thread_count = config['decoder_threads']
        for f, frame in enumerate(container.decode(video=0)):
            if f >= limit:
                break
            decoded += 1
            if f in frame_requests:
                image = frame.to_ndarray(format='rgb24')
                if image.shape[:2] != (mapper.video_height, mapper.video_width):
                    raise ValueError('Decoded dimensions changed')
                suspect = border_padding_suspect(image, config['black_threshold'])
                frame_hashes[str(f)] = hashlib.sha256(image.tobytes()).hexdigest()
                for row in frame_requests[f]:
                    patch = supported_center_patch(image, (boxes[row, :2]+boxes[row, 2:])/2,
                        suspect, config['crop_size'], size)
                    for name, value in patch.items():
                        arrays[name][row] = value
                if f in review_frames:
                    images[f] = image
            if f % 1000 == 0:
                heartbeat(output, phase='extracting_registered_past_pixels', recording=key,
                          frame=f, requested_frame_count=len(frame_requests), elapsed=time.monotonic()-start)
            if f == limit-1:
                break
    if decoded != limit or len(frame_hashes) != len(frame_requests):
        raise ValueError('Video does not cover registered requests; do not drop queries')
    for name in ARRAYS:
        temporary = directory/(name+'.tmp.npy')
        np.save(temporary, arrays[name], allow_pickle=False)
        os.replace(temporary, directory/(name+'.npy'))
    metadata = dict(schema_version=1, data_role='diagnostic_only', training_admitted=False,
        observation_mode='offline_annotated', crop_size=config['crop_size'], output_size=size,
        decoded_prefix_frames=limit, cache_scope='sparse_registered_step_history_requests_not_dense_prefix',
        array_sha256={name:file_digest(directory/(name+'.npy')) for name in ARRAYS},
        source=entry, identity=identity, future_targets_in_cache=False,
        padding_annotation_quality='inferred_only_not_person_visibility', strict_sensor_as_of_certified=False)
    json_write(directory/'metadata.json', metadata); json_write(directory/'queries.json', queries)
    store = SDDStepImageStore(directory, observation_mode='offline_annotated')
    joined = check_model_join(store, adapters, queries)
    save_review(directory, store, queries, images)
    geometric = arrays['geometric_count'].sum((1, 2), dtype=np.int64)
    retained = arrays['retained_count'].sum((1, 2), dtype=np.int64)
    receipt = dict(identity=identity, result_source='fresh_run', recording=key,
        source_rows=len(rows), geometry_join=joined, decoded_frames=decoded,
        requested_unique_frames=len(frame_requests), requested_frame_agent_rows=len(chosen),
        query_frame_range=[min(q['frame_id'] for q in queries), max(q['frame_id'] for q in queries)],
        requested_rows_after_frame63=int((chosen[:, 5] > 63).sum()),
        source_generated_rows=int(chosen[:, 8].sum()), source_occluded_rows=int(chosen[:, 7].sum()),
        geometric_partial_rows=int((geometric < config['crop_size']**2).sum()),
        rows_with_suspect_padding=int((retained < geometric).sum()),
        no_retained_pixel_rows=int((retained == 0).sum()),
        sampled_frame_sha256=frame_hashes, cache_array_bytes=sum((directory/(name+'.npy')).stat().st_size for name in ARRAYS),
        elapsed_seconds=time.monotonic()-start, visual_review_status='not_reviewed',
        artifacts={name:file_digest(directory/name) for name in ('metadata.json', 'queries.json', 'private_contact_sheet.png')})
    json_write(receipt_path, receipt)
    return receipt, True


def verify_record(entry, config, identity, av, output):
    directory = output/entry['annotation_key']
    receipt = json.loads((directory/'receipt.json').read_text())
    if receipt['identity'] != identity:
        raise ValueError('Changed replay identity')
    if file_digest(ROOT/entry['video_path']) != entry['video_sha256']:
        raise ValueError('Changed source video')
    store = SDDStepImageStore(directory, observation_mode='offline_annotated')
    frames = sorted(map(int, receipt['sampled_frame_sha256']))
    selected = {frames[i] for i in np.unique([0, len(frames)//2, len(frames)-1])}
    patch_checks = frame_checks = 0
    with av.open(str(ROOT/entry['video_path'])) as container:
        container.streams.video[0].codec_context.thread_count = config['decoder_threads']
        for f, frame in enumerate(container.decode(video=0)):
            if f in selected:
                image = frame.to_ndarray(format='rgb24')
                if hashlib.sha256(image.tobytes()).hexdigest() != receipt['sampled_frame_sha256'][str(f)]:
                    raise ValueError('Independent decoded pixels differ')
                suspect = border_padding_suspect(image, config['black_threshold'])
                for row in np.flatnonzero(store.arrays['row_keys'][:, 0] == f):
                    box = store.arrays['image_boxes'][row]
                    patch = supported_center_patch(image, (box[:2]+box[2:])/2, suspect, config['crop_size'], config['output_size'])
                    for name, value in patch.items():
                        np.testing.assert_array_equal(value, store.arrays[name][row])
                    patch_checks += 1
                frame_checks += 1
            if f >= max(selected):
                break
    if frame_checks != len(selected):
        raise ValueError('Missing replay frames')
    return dict(recording=entry['annotation_key'], independent_decoded_frames=frame_checks,
                exact_patch_replays=patch_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--recording')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['training_admitted'] or config['data_role'] != 'diagnostic_only':
        raise ValueError('No training or new source role admitted')
    for name, digest in config['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Bound artifact changed: '+name)
    bridge = json.loads((ROOT/config['bridge_registration']).read_text())
    for name, digest in bridge['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Geometry bridge changed: '+name)
    links = json.loads((ROOT/bridge['source_manifest']).read_text())
    completed = json.loads((ROOT/bridge['reports']/'report.json').read_text())
    ids = {r['recording'] for r in completed['recordings']}
    chosen = [e for e in links['records'] if e['annotation_key'] in ids]
    if len(chosen) != 40 or (args.recording and args.recording not in ids):
        raise ValueError('Only original 40 train recordings admitted')
    sys.path.insert(0, str(ROOT/config['decoder_path']))
    import av
    if av.__version__ != config['decoder_version']:
        raise ValueError('Decoder version changed')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    identity = dict(config_sha256=file_digest(args.config), torch_version=torch.__version__,
                    numpy_version=np.__version__, av_version=av.__version__)
    output = ROOT/config['output']; reports = ROOT/config['reports']
    receipts, fresh, reused = [], 0, 0
    for entry in chosen:
        if args.recording and args.recording != entry['annotation_key']:
            continue
        if args.verify:
            record = verify_record(entry, config, identity, av, output)
            receipts.append(record)
            print(json.dumps(record), flush=True)
        else:
            record, new = build_record(entry, config, bridge, identity, av, output)
            receipts.append(record); fresh += int(new); reused += int(not new)
            print(json.dumps(dict(recording=entry['annotation_key'], fresh=new,
                joined=record['geometry_join'], seconds=record['elapsed_seconds'])), flush=True)
    if not args.recording:
        name = 'verification.json' if args.verify else 'report.json'
        public_records = [{k:v for k,v in r.items() if k != 'sampled_frame_sha256'} for r in receipts]
        result = dict(identity=identity, result_source='fresh_run', records=public_records,
            data_role='diagnostic_only', source_role_and_training_stride_pending=True,
            original_train_recordings=len(receipts), original_val_test_opened=False,
            new_model_training=False, optimizer_updates=0, predictive_gain_measured=False,
            stage5c_executed=False, smc_enabled=False)
        if (reports/name).exists():
            if json.loads((reports/name).read_text()) != result:
                raise ValueError('Completed report changed')
        else:
            json_write(reports/name, result)
    heartbeat(output, phase='complete_verification' if args.verify else 'complete',
              pilot=bool(args.recording), fresh=fresh, reused=reused, new_training=False)
    print(json.dumps(dict(complete=True, pilot=bool(args.recording), verify=args.verify, fresh=fresh, reused=reused)), flush=True)


if __name__ == '__main__':
    main()
