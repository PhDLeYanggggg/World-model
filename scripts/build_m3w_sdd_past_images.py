"""Build or replay a fixed-prefix SDD past-image diagnostic, without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from PIL import Image, ImageDraw

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_image_coordinates import SDDImageCoordinates
from src.world_model.m3w_sdd_media_audit import annotation_summary
from src.world_model.m3w_sdd_past_images import (
    ARRAYS, SDDPastImageStore, border_padding_suspect, supported_center_patch,
)


CONFIG = ROOT/'configs/m3w_sdd_past_image_audit.json'


def heartbeat(output, **values):
    json_write(output/'heartbeat.json', dict(pid=os.getpid(), updated_unix=time.time(), **values))


def validate_sources(entry):
    for name in ('annotations', 'reference', 'video'):
        if file_digest(ROOT/entry[name+'_path']) != entry[name+'_sha256']:
            raise ValueError('Changed bound source '+entry['annotation_key']+': '+name)


def prefix_rows(entry, limit):
    all_rows = np.loadtxt(ROOT/entry['annotations_path'], usecols=range(9), dtype=float, ndmin=2)
    rows = all_rows[all_rows[:, 5] < limit]
    rows = rows[np.lexsort((rows[:, 0], rows[:, 5]))]
    if len(rows):
        summary = annotation_summary(rows, *entry['geometry']['annotation_size'])
        if summary['duplicate_agent_frame_rows']:
            raise ValueError('Duplicate past agent/frame rows')
    return rows


def canvas_frame(image, suspect, arrays, frame_id, entry):
    """Private audit visualization, chosen by input support, never forecast error."""
    overlay = image.copy()
    overlay[suspect] = [255, 60, 60]
    panel = Image.new('RGB', (720, 400), 'white')
    for col, pixels in enumerate((image, overlay)):
        thumb = Image.fromarray(pixels)
        thumb.thumbnail((350, 255))
        panel.paste(thumb, (col*360, 25))
    draw = ImageDraw.Draw(panel)
    draw.text((3, 3), entry['annotation_key']+' frame '+str(frame_id), fill='black')
    draw.text((362, 3), 'red: inferred border-connected dark area', fill='black')
    indices = np.flatnonzero((arrays['row_keys'][:, 0] == frame_id) & (arrays['source_flags'][:, 0] == 0))
    if len(indices):
        removed = (arrays['geometric_count'][indices].astype(int)-arrays['retained_count'][indices]).sum((1, 2))
        row = indices[np.argmax(removed)]
        draw.text((3, 283), 'Input-selected example, agent '+str(arrays['row_keys'][row, 1]), fill='black')
        for col, name in enumerate(('rgb_observed', 'rgb_retained', 'geometric_count', 'retained_count')):
            pixels = arrays[name][row]
            pixels = pixels.transpose(1, 2, 0) if pixels.ndim == 3 else (pixels*(255/9)).astype(np.uint8)
            thumb = Image.fromarray(pixels).resize((80, 80), Image.Resampling.NEAREST)
            panel.paste(thumb, (col*180, 320))
            draw.text((col*180, 303), name, fill='black')
    return panel


def window_audit(store, reg):
    records = []
    for query in reg['query_frames']:
        if query >= store.prefix:
            continue
        agents = store.agents_at(query)
        for length in reg['history_lengths']:
            item = dict(query_frame=query, length=length, agents=len(agents),
                        complete_annotation_histories=0, complete_nonlost_histories=0,
                        complete_retained_image_histories=0, no_retained_image_histories=0)
            for agent in agents:
                a = store.inputs(query_frame=query, agent_id=int(agent), length=length,
                                 step=reg['history_frame_step'])
                item['complete_annotation_histories'] += int(a['annotation_present_mask'].all())
                item['complete_nonlost_histories'] += int(a['state_mask'].all())
                item['complete_retained_image_histories'] += int(a['image_frame_mask'].all())
                item['no_retained_image_histories'] += int(not a['image_frame_mask'].any())
            records.append(item)
    return records


def build_record(entry, reg, identity, output, av):
    directory = output/entry['annotation_key']
    receipt_path = directory/'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['identity'] != identity:
            raise ValueError('Changed source/code identity; use a new version, not overwrite')
        store = SDDPastImageStore(directory, observation_mode='offline_annotated')
        if file_digest(directory/'metadata.json') != receipt['metadata_sha256']:
            raise ValueError('Changed cache metadata')
        if receipt['windows'] != window_audit(store, reg):
            raise ValueError('Changed window replay')
        return receipt, False
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    limit = min(reg['prefix_frames'], entry['coverage']['decoded_frames'])
    rows = prefix_rows(entry, limit)
    n, size = len(rows), reg['output_size']
    mapper = SDDImageCoordinates(*entry['geometry']['annotation_size'], *entry['geometry']['video_size'])
    boxes = mapper.past_boxes(rows[:, 1:5], frame_ids=rows[:, 5], query_frame=limit-1)
    arrays = dict(row_keys=rows[:, [5, 0]].astype(np.int64), annotation_boxes=rows[:, 1:5],
                  image_boxes=boxes, source_flags=rows[:, 6:9].astype(np.uint8))
    for name in ('rgb_observed', 'rgb_retained'):
        arrays[name] = np.zeros((n, 3, size, size), np.uint8)
    for name in ('geometric_count', 'retained_count'):
        arrays[name] = np.zeros((n, size, size), np.uint8)
    requests = {f: np.flatnonzero(rows[:, 5] == f) for f in range(limit)}
    decoded, padding_fractions, pixel_hashes, panels = [], [], {}, []
    with av.open(str(ROOT/entry['video_path'])) as container:
        container.streams.video[0].codec_context.thread_count = reg['decoder_threads']
        for frame_id, frame in enumerate(container.decode(video=0)):
            if frame_id >= limit:
                break
            image = frame.to_ndarray(format='rgb24')
            if image.shape[:2] != (mapper.video_height, mapper.video_width):
                raise ValueError('Changed decoded dimensions')
            suspect = border_padding_suspect(image, reg['black_threshold'])
            decoded.append(frame_id)
            padding_fractions.append(float(suspect.mean()))
            pixel_hashes[str(frame_id)] = hashlib.sha256(image.tobytes()).hexdigest()
            for row in requests[frame_id]:
                if arrays['source_flags'][row, 0]:
                    continue
                center = (boxes[row, :2]+boxes[row, 2:])/2
                patch = supported_center_patch(image, center, suspect, reg['crop_size'], size)
                for name, value in patch.items():
                    arrays[name][row] = value
            if frame_id in (0, 31, 63):
                panels.append(canvas_frame(image, suspect, arrays, frame_id, entry))
            if frame_id % 16 == 0:
                heartbeat(output, phase='decoding_past_images', recording=entry['annotation_key'], frame=frame_id)
            if frame_id == limit-1:
                break
    if decoded != list(range(limit)):
        raise ValueError('Missing decoded prefix frame')
    for name in ARRAYS:
        np.save(directory/(name+'.npy'), arrays[name], allow_pickle=False)
    metadata = dict(schema_version=1, data_role='diagnostic_only', training_admitted=False,
        observation_mode='offline_annotated', crop_size=reg['crop_size'], output_size=size,
        decoded_prefix_frames=limit, array_sha256={n:file_digest(directory/(n+'.npy')) for n in ARRAYS},
        source_manifest_sha256=reg['source_manifest_sha256'], source=entry, identity=identity,
        padding_annotation_quality=reg['padding_annotation_quality'],
        coordinate_unit='annotation_pixel', generated_annotation_provenance_preserved=True,
        pixel_mask_is_not_person_visibility=True, strict_sensor_as_of_certified=False,
        future_targets_in_cache=False, population_requires_nonlost_observation_at_or_before_query=True)
    json_write(directory/'metadata.json', metadata)
    store = SDDPastImageStore(directory, observation_mode='offline_annotated')
    visible = arrays['source_flags'][:, 0] == 0
    g = arrays['geometric_count'].sum((1, 2), dtype=np.int64)
    r = arrays['retained_count'].sum((1, 2), dtype=np.int64)
    sheet = Image.new('RGB', (720, 400*len(panels)), 'white')
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, 400*i))
    sheet.save(directory/'private_support_review.png')
    receipt = dict(identity=identity, result_source='fresh_run_past_pixel_extraction',
        recording=entry['annotation_key'], rows=n, decoded_frames=limit,
        visible_rows=int(visible.sum()), lost_rows=int((~visible).sum()),
        occluded_rows=int(arrays['source_flags'][:, 1].sum()),
        generated_rows=int(arrays['source_flags'][:, 2].sum()),
        visible_partial_bounds_rows=int((visible & (g < reg['crop_size']**2)).sum()),
        visible_rows_with_suspect_border=int((visible & (r < g)).sum()),
        visible_rows_no_retained_pixels=int((visible & (r == 0)).sum()),
        suspected_border_frame_fraction_quantiles=np.quantile(padding_fractions, [0, .5, 1]).tolist(),
        suspected_border_frame_fractions=padding_fractions, decoded_frame_pixel_sha256=pixel_hashes,
        windows=window_audit(store, reg), metadata_sha256=file_digest(directory/'metadata.json'),
        cache_array_bytes=sum((directory/(n+'.npy')).stat().st_size for n in ARRAYS),
        elapsed_seconds=time.monotonic()-started,
        private_visual_sha256=file_digest(directory/'private_support_review.png'), visual_review_status='not_reviewed')
    json_write(receipt_path, receipt)
    return receipt, True


def verify_record(entry, reg, identity, output, av):
    directory = output/entry['annotation_key']
    receipt = json.loads((directory/'receipt.json').read_text())
    if receipt['identity'] != identity or file_digest(directory/'metadata.json') != receipt['metadata_sha256']:
        raise ValueError('Changed extraction identity')
    store = SDDPastImageStore(directory, observation_mode='offline_annotated')
    rows = prefix_rows(entry, store.prefix)
    np.testing.assert_array_equal(store.arrays['row_keys'], rows[:, [5, 0]].astype(np.int64))
    np.testing.assert_array_equal(store.arrays['annotation_boxes'], rows[:, 1:5])
    np.testing.assert_array_equal(store.arrays['source_flags'], rows[:, 6:9].astype(np.uint8))
    mapper = SDDImageCoordinates(*entry['geometry']['annotation_size'], *entry['geometry']['video_size'])
    mapped = mapper.past_boxes(rows[:, 1:5], frame_ids=rows[:, 5], query_frame=store.prefix-1)
    np.testing.assert_array_equal(store.arrays['image_boxes'], mapped)
    patch_checks, frame_checks, mutation_checks = 0, 0, 0
    with av.open(str(ROOT/entry['video_path'])) as container:
        container.streams.video[0].codec_context.thread_count = reg['decoder_threads']
        for f, frame in enumerate(container.decode(video=0)):
            if f in (0, 31, 63) and f < store.prefix:
                image = frame.to_ndarray(format='rgb24')
                if hashlib.sha256(image.tobytes()).hexdigest() != receipt['decoded_frame_pixel_sha256'][str(f)]:
                    raise ValueError('Independent decoded pixels differ')
                frame_checks += 1
                suspect = border_padding_suspect(image, reg['black_threshold'])
                indices = np.flatnonzero((rows[:, 5] == f) & (rows[:, 6] == 0))
                for index in indices:
                    patch = supported_center_patch(image, (mapped[index, :2]+mapped[index, 2:])/2,
                                                   suspect, reg['crop_size'], reg['output_size'])
                    for name in patch:
                        np.testing.assert_array_equal(patch[name], store.arrays[name][index])
                    patch_checks += 1
            if f >= store.prefix-1:
                break
    original = store.arrays
    for q in (0, 7, 31):
        agents = store.agents_at(q)
        if not len(agents):
            continue
        before = store.inputs(query_frame=q, agent_id=int(agents[0]), length=64)
        store.arrays = {name: np.array(value) for name, value in original.items()}
        future = original['row_keys'][:, 0] > q
        for name in ARRAYS:
            if name != 'row_keys':
                store.arrays[name][future] = 0
        store.arrays['row_keys'][future, 1] += 100000
        after = store.inputs(query_frame=q, agent_id=int(agents[0]), length=64)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
        np.testing.assert_array_equal(agents, store.agents_at(q))
        store.arrays = original
        mutation_checks += 1
    return dict(recording=entry['annotation_key'], independently_redecoded_frames=frame_checks,
                exact_patch_replays=patch_checks, future_row_mutation_checks=mutation_checks,
                source_rows_and_geometry_exact=True, cache_sha256=receipt['metadata_sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--recording', help='Source diagnostic pilot; does not change the registered full inventory')
    args = parser.parse_args()
    reg = json.loads(CONFIG.read_text())
    if (reg['training_admitted'] or reg['model_fitting_allowed'] or reg['sealed_evaluation_access_allowed']
            or reg['data_role'] != 'diagnostic_only'):
        raise ValueError('Source diagnostic cannot admit a scientific role')
    if file_digest(ROOT/reg['source_manifest']) != reg['source_manifest_sha256']:
        raise ValueError('Changed media correspondence')
    manifest = json.loads((ROOT/reg['source_manifest']).read_text())
    if manifest['new_training_source_admitted'] or manifest['official_eval_allowed']:
        raise ValueError('Unexpected manifest admission')
    sys.path.insert(0, str(ROOT/reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Changed decoder')
    identity = dict(config_sha256=file_digest(CONFIG), decoder_version=av.__version__,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        code_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in (Path(__file__),
            ROOT/'src/world_model/m3w_sdd_past_images.py', ROOT/'src/world_model/m3w_sdd_image_coordinates.py')})
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    output.mkdir(parents=True, exist_ok=True)
    entries = [r for r in manifest['records'] if args.recording is None or r['annotation_key'] == args.recording]
    if not entries:
        raise ValueError('No matching recording')
    records, fresh = [], 0
    for entry in entries:
        validate_sources(entry)
        if args.verify:
            current = verify_record(entry, reg, identity, output, av)
        else:
            current, rebuilt = build_record(entry, reg, identity, output, av)
            fresh += int(rebuilt)
        records.append(current)
        print(json.dumps(dict(recording=entry['annotation_key'], completed=len(records), total=len(entries),
                              phase='verified' if args.verify else 'built_or_verified_resume')), flush=True)
    result = dict(identity=identity, records=records, complete_inventory=len(records) == len(manifest['records']),
        result_source='fresh_run_independent_sample_decode_and_cached_hash_checks' if args.verify
            else 'fresh_run_extraction_and_hash_verified_resume',
        new_recordings_extracted_this_invocation=fresh, model_trained=False,
        source_role_admitted=False, sealed_evaluation_roles_opened=False,
        metric_or_seconds_claim=False, stage5c_executed=False, smc_enabled=False)
    suffix = '' if args.recording is None else '_'+args.recording.replace('/', '_')
    json_write(reports/(('verification' if args.verify else 'audit')+suffix+'.json'), result)
    heartbeat(output, phase='complete', recordings=len(records), complete_inventory=result['complete_inventory'])


if __name__ == '__main__':
    main()
