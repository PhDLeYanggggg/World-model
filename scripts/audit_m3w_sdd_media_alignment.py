"""Decode every local SDD frame and audit annotations, with per-video resume."""
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
from PIL import Image, ImageDraw
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_media_audit import annotation_summary, decoded_coverage, reference_comparison


def evidence_paths(record):
    video = ROOT/record['path']
    suffix = Path(record['scene'])/record['video']
    return dict(video=video,
        annotations=ROOT/'external_data/StanfordDroneDataset/annotations'/suffix/'annotations.txt',
        reference=ROOT/'external_data/StanfordDroneDataset/annotations'/suffix/'reference.jpg',
        mirror=ROOT/'external_data/OpenTraj/datasets/SDD'/suffix/'annotations.txt')


def contact_sheet(images, rows, destination, key):
    canvas = Image.new('RGB', (480*len(images), 510), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, (index, image) in enumerate(sorted(images.items())):
        overlay = Image.fromarray(image.copy())
        pen = ImageDraw.Draw(overlay)
        current = rows[(rows[:, 5] == index) & (rows[:, 6] == 0)]
        for row in current:
            pen.rectangle(tuple(row[1:5]), outline=(230, 40, 50), width=2)
        overlay.thumbnail((480, 475))
        canvas.paste(overlay, (j*480, 28))
        draw.text((j*480+4, 5), f'{key} decoded frame {index}; visible boxes {len(current)}', fill='black')
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', help='Optional pilot scene/video, then full run resumes it')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    inventory_path = ROOT/'outputs/publication_readiness_2026_09/sdd_media_inventory/headers.json'
    inventory = json.loads(inventory_path.read_text())
    parent = json.loads((ROOT/'configs/m3w_offline_visual_forecast.json').read_text())
    sys.path.insert(0, str(ROOT/parent['decoder_path']))
    import av
    output = ROOT/'data/stage_cvpr2027_experiments/sdd_media_alignment'
    reports = ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment'
    output.mkdir(parents=True, exist_ok=True)
    code_identity = {str(path.relative_to(ROOT)): file_digest(path) for path in
        (Path(__file__), ROOT/'src/world_model/m3w_sdd_media_audit.py')}
    audit_identity = dict(inventory_sha256=file_digest(inventory_path), code_sha256=code_identity,
        decoder_version=av.__version__, decoder_threads=4, data_role='source_audit_only')

    def heartbeat(**items):
        item = dict(pid=os.getpid(), time_unix=time.time(), **items)
        json_write(output/'heartbeat.json', item)
        print(json.dumps(item), flush=True)

    results = []
    new_decodes = 0
    for record in inventory['records']:
        key = record['scene']+'/'+record['video']
        if args.recording and args.recording != key:
            continue
        paths = evidence_paths(record)
        hashes = {name: file_digest(path) if path.exists() else None for name, path in paths.items()}
        if hashes['video'] != record['sha256'] or not hashes['annotations']:
            raise ValueError('Media inventory changed or annotations missing: '+key)
        identity = dict(audit_identity=audit_identity, source_hashes=hashes, key=key)
        saved = output/'records'/(key.replace('/', '_')+'.json')
        if saved.exists():
            cached = json.loads(saved.read_text())
            if cached['identity'] != identity:
                raise ValueError('Changed audit/source; register a new run, do not overwrite '+key)
            if file_digest(output/cached['contact_sheet']) != cached['contact_sheet_sha256']:
                raise ValueError('Changed private visual evidence: '+key)
            results.append(cached)
            heartbeat(state='cached_verified', recording=key, completed=len(results))
            continue
        if args.verify_only:
            raise ValueError('Missing completed decode: '+key)
        started = time.perf_counter()
        rows = np.loadtxt(paths['annotations'], usecols=range(9), dtype=float, ndmin=2)
        annotation = annotation_summary(rows, record['width'], record['height'])
        wanted = set(np.linspace(0, record['header_frames']-1, 5, dtype=int).tolist())
        images = {}
        count, pts_missing, pts_nonincreasing = 0, 0, 0
        previous_pts, first_pts, last_pts = None, None, None
        dimension_mismatch = 0
        last_heartbeat = time.perf_counter()
        decode_hash = hashlib.sha256()
        heartbeat(state='decoding', recording=key, frames=0, completed=len(results))
        with av.open(str(paths['video'])) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 4
            for frame in container.decode(video=0):
                if frame.width != record['width'] or frame.height != record['height']:
                    dimension_mismatch += 1
                if frame.pts is None:
                    pts_missing += 1
                else:
                    if previous_pts is not None and frame.pts <= previous_pts:
                        pts_nonincreasing += 1
                    if first_pts is None:
                        first_pts = int(frame.pts)
                    previous_pts = last_pts = int(frame.pts)
                decode_hash.update(f'{count},{frame.pts},{frame.width},{frame.height};'.encode('ascii'))
                if count in wanted:
                    images[count] = frame.to_ndarray(format='rgb24')
                count += 1
                if time.perf_counter()-last_heartbeat >= 10:
                    heartbeat(state='decoding', recording=key, frames=count,
                              seconds=time.perf_counter()-started, completed=len(results))
                    last_heartbeat = time.perf_counter()
        reference = (reference_comparison(images[0], np.asarray(Image.open(paths['reference']).convert('RGB')))
                     if paths['reference'].exists() else dict(missing=True))
        sheet = Path('private_visual_checks')/(key.replace('/', '_')+'.png')
        contact_sheet(images, rows, output/sheet, key)
        result = dict(identity=identity, result_source='fresh_run_full_decode_and_index_audit',
            annotation=annotation, coverage=decoded_coverage(rows[:, 5], count, record['header_frames']),
            supplied_mirror_annotation_hash_match=bool(hashes['annotations'] == hashes['mirror']),
            decoded_dimensions_mismatching_header=dimension_mismatch,
            pts_missing=pts_missing, pts_nonincreasing=pts_nonincreasing,
            first_pts=first_pts, last_pts=last_pts, decode_index_pts_dimensions_sha256=decode_hash.hexdigest(),
            sampled_frame_indices=sorted(images), sampled_frame_pixel_sha256={str(k):hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()},
            reference_first_frame=reference, contact_sheet=str(sheet), contact_sheet_sha256=file_digest(output/sheet),
            full_pixel_checksum_not_computed=True, seconds=time.perf_counter()-started,
            source_training_admitted=False, forecast_targets_constructed=False,
            feature_store_or_model_built=False, sealed_evaluation_roles_opened=False)
        json_write(saved, result)
        results.append(result)
        new_decodes += 1
        heartbeat(state='record_complete', recording=key, frames=count,
                  seconds=result['seconds'], completed=len(results))
    if not results:
        raise ValueError('No selected recording')
    if args.recording:
        heartbeat(state='pilot_complete', recording=args.recording, new_decodes=new_decodes)
        return
    aggregate = dict(identity=audit_identity, records=results, complete=len(results)==len(inventory['records']),
        result_source='fresh_run_full_decode_with_per_video_resume',
        videos=len(results), frames=sum(r['coverage']['decoded_frames'] for r in results),
        annotation_rows=sum(r['annotation']['rows'] for r in results),
        header_decode_matches=sum(r['coverage']['header_matches_decode'] for r in results),
        annotation_index_coverage_pass=sum(r['coverage']['decoded_range_covers_annotations'] for r in results),
        supplied_mirror_hash_matches=sum(r['supplied_mirror_annotation_hash_match'] for r in results),
        summed_video_audit_seconds=sum(r['seconds'] for r in results),
        video_id_is_not_independent_scene=True, historical_dataset_exposure_retained=True,
        new_training_source_admitted=False, stage5c_executed=False, smc_enabled=False,
        physical_time_or_metric_certified=False, semantic_alignment_certified=False)
    report_path = reports/'audit.json'
    if report_path.exists():
        if json.loads(report_path.read_text()) != aggregate:
            raise ValueError('Completed audit differs; preserve original report')
    else:
        json_write(report_path, aggregate)
    if args.verify_only:
        json_write(reports/'verification.json', dict(result_source='cached_verified_hash_and_source_identity',
            report_sha256=file_digest(report_path), verified_records=len(results), new_decodes=new_decodes,
            second_full_decode=False))
    heartbeat(state='verified' if args.verify_only else 'complete', records=len(results), new_decodes=new_decodes)


if __name__ == '__main__':
    main()
