"""Pinned past-image measurement on producer-A only; no future target arrays."""
import argparse
import fcntl
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import zipfile

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from scripts import intake_m3w_movingdronecrowd as intake
from src.data_unification import m3w_past_background_motion as method
from src.data_unification.m3w_movingdronecrowd_intake import PublicRangeZip, clip_key

NAME = 'moving_past_background_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs'/f'm3w_{NAME}.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(method.__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'tests/test_m3w_past_background_motion.py')
PAST_KEYS = ('sites', 'recordings', 'frames', 'queries', 'global_ids', 'agents', 'history')
sha, once, atomic = intake.sha, intake.once, intake.atomic


def read(path):
    return json.loads(Path(path).read_text())


def stable(x):
    return json.loads(json.dumps(x, sort_keys=True, allow_nan=False))


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path))


def checked(ref):
    p = (ROOT/ref['path']).resolve()
    if not p.is_relative_to(ROOT) or sha(p) != ref['sha256']:
        raise ValueError('Changed artifact')
    return p


def past_arrays(cfg):
    manifest = read(ROOT/cfg['parent_population']); data = {}
    for key in PAST_KEYS:
        data[key] = np.load(checked(manifest['files'][key]), mmap_mode='r', allow_pickle=False)
    if len(data['sites']) != cfg['expected_rows'] or data['history'].shape != (cfg['expected_rows'], 8, 2):
        raise ValueError('Retain the complete past-only population')
    return data


def plan(cfg):
    data = past_arrays(cfg); admission = read(ROOT/cfg['admission'])
    names = sorted(admission['recordings']); parent = read(ROOT/cfg['parent_registration'])
    allowed = {s for j in parent['jobs'] for s in j['producer_sites']}
    if set(data['sites']) != allowed or len(allowed) != cfg['expected_sources']:
        raise ValueError('Existing A-only scope required')
    inventory = read(ROOT/cfg['archive_inventory'])
    images = {}
    for item in inventory:
        clip = clip_key(item['name'], 'frames')
        if clip is not None:
            images[clip, int(Path(item['name']).stem)] = item
    records = []; frames = []
    for index in sorted(set(data['recordings'])):
        node = names[int(index)]; meta = admission['recordings'][node]
        at = data['recordings'] == index; queries = np.unique(data['frames'][at])
        if set(data['sites'][at]) != {meta['exclusion_group']} or queries.min() < 7:
            raise ValueError('Causal frame/source mapping changed')
        body = ROOT/meta['source_annotation_path']
        if sha(body) != meta['source_annotation_sha256']:
            raise ValueError('Changed annotation bridge')
        records.append(dict(node=node, recording_index=int(index), source=meta['exclusion_group'],
            clip=meta['clip'], query_frames=queries.tolist(), rows=int(at.sum()), annotation=artifact(body),
            original_image_inputs_admitted=meta['image_inputs_admitted']))
        for frame in range(int(queries.max())+1):
            item = images[meta['clip'], frame+1]
            frames.append(dict(node=node, annotation_frame=frame, image_frame=frame+1, member=item))
    queries = sum(len(r['query_frames']) for r in records)
    if (len(records) != cfg['expected_recordings'] or queries != cfg['expected_queries']
            or len(frames) != cfg['expected_images'] or len(frames)-len(records) != cfg['expected_pairs']
            or sum(f['member']['compressed_bytes'] for f in frames) != cfg['expected_compressed_member_bytes']):
        raise ValueError('Fixed full source roster required')
    return dict(records=records, frames=frames, rows=len(data['sites']), queries=queries,
        image_members=len(frames), transitions=len(frames)-len(records), future_target_arrays_opened=False)


def register():
    cfg = read(CONFIG); source_cfg = read(ROOT/cfg['source_config'])
    if (cfg['image_scope'] != 'existing_producer_A_past_only_measurement' or cfg['expected_rows'] != 125540
            or len(method.NAMES) != 48 or cfg['num_workers'] != 0
            or any(cfg[k] for k in ('retain_raw_JPEG', 'predictive_training_in_this_measurement',
                'future_target_arrays_read', 'future_targets_in_features', 'new_own_B_readout',
                'new_outer_C_H_readout', 'identity_certification', 'deployment_changed', 'submission_ready',
                'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Measurement-only A scope and unchanged original rules required')
    source = intake.cached_source(); roster = plan(cfg)
    population = read(ROOT/cfg['parent_population'])
    value = stable(dict(config=cfg, files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in
            (Path(intake.__file__), ROOT/'src/data_unification/m3w_movingdronecrowd_intake.py')},
        parents=[artifact(ROOT/cfg[k]) for k in ('parent_population', 'parent_registration', 'admission',
            'source_config', 'source_manifest', 'archive_inventory')],
        roster=roster, past_arrays={k: population['files'][k] for k in PAST_KEYS},
        archive_url=source['archive_url'], source_config=source_cfg, feature_names=list(method.NAMES),
        author_reader=artifact(ROOT/'data/stage_cvpr2027_experiments/movingdronecrowd_intake_v1/publisher_dataset_reader.py'),
        previous_negative=artifact(ROOT/'outputs/publication_readiness_2026_09/moving_prefix_actor_v1/analysis.json'),
        opencv=cv2.__version__, numpy=np.__version__, machine=platform.machine(),
        original_rules_unchanged=True, original_admission_unchanged=True, new_predictive_admission=False))
    once(REG, value); return cfg, value


def committed(reg):
    for p in (*FILES, REG):
        b = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
            capture_output=True, check=True).stdout
        if hashlib.sha256(b).hexdigest() != sha(p):
            raise ValueError('Commit exact measurement before acquiring pixels')
    for p, digest in reg['dependencies'].items():
        if sha(ROOT/p) != digest:
            raise ValueError('Changed pinned range/intake implementation')


def runtime(cfg, phase):
    cv2.setNumThreads(cfg['opencv_threads']); cv2.ocl.setUseOpenCL(False)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic image/pair receipts')))
    start = time.monotonic()
    def beat(**values):
        seconds = time.monotonic()-start
        size = sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
        if seconds > cfg['maximum_invocation_seconds']:
            raise TimeoutError('Resume unchanged measurement after12h boundary')
        if size > cfg['maximum_cache_bytes'] or shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Keep registered cache limit and5GiB reserve')
        event = dict(phase=phase, pid=os.getpid(), seconds=seconds,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), private_bytes=size,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **values)
        atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    beat(state='ready', device='cpu', machine=platform.machine(), opencv_threads=4, workers=0, torch_resource_probing=False)
    return start, beat


def frame_path(row):
    return PRIVATE/'rasters'/row['node']/f'{row["annotation_frame"]:06d}.npz'


@lru_cache(maxsize=64)
def annotation_index(path, digest):
    return method.boxes_by_frame(checked(dict(path=path, sha256=digest)).read_bytes())


def cache_frame(row, reg, archive, cfg):
    path = frame_path(row); receipt = path.with_suffix('.json')
    if receipt.exists():
        meta = read(receipt); checked(meta['cache'])
        if meta['registration_sha256'] != sha(REG) or meta['row'] != row:
            raise ValueError('Changed image cache identity')
        return meta, False
    raw = archive.read(row['member']['name'])
    if len(raw) != row['member']['bytes'] or len(raw) > cfg['maximum_raw_member_bytes']:
        raise ValueError('Changed or oversized image member')
    with Image.open(io.BytesIO(raw)) as image:
        if image.width*image.height > cfg['maximum_image_pixels']:
            raise ValueError('Oversized native image')
        orientation = int(image.getexif().get(274, 1)); rgb = np.asarray(image.convert('RGB'))
    record = next(r for r in reg['roster']['records'] if r['node'] == row['node'])
    indexed = annotation_index(record['annotation']['path'], record['annotation']['sha256'])
    prepared = method.prepare(rgb, indexed.get(row['annotation_frame'], np.empty((0, 4))), cfg['measurement'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # A crash between two atomic saves can leave a raster without its receipt.
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(prepared):
                raise ValueError('Unreceipted raster schema differs')
            for key in prepared:
                np.testing.assert_array_equal(z[key], prepared[key])
    else:
        pending = path.with_suffix('.pending')
        with pending.open('wb') as f:
            np.savez_compressed(f, **prepared)
        pending.replace(path)
    value = stable(dict(registration_sha256=sha(REG), row=row, cache=artifact(path),
        JPEG_sha256=hashlib.sha256(raw).hexdigest(), ZIP_CRC_checked=True,
        native_size=prepared['native_size'].tolist(), resize_scale=prepared['resize_scale'].tolist(),
        native_centers=prepared['centers'], native_centers_in_bounds=prepared['centers_in_bounds'],
        exif_orientation=orientation, PIL_native_axes_preserved=True, raw_JPEG_retained=False))
    once(receipt, value); return value, True


def acquire(cfg, reg, rows, beat, resume):
    present = any((PRIVATE/'rasters').glob('*/*.json'))
    if present and not resume:
        raise ValueError('Explicit unchanged image acquisition --resume required')
    source_cfg = reg['source_config']
    api = f'https://huggingface.co/api/datasets/{source_cfg["repository"]}/revision/{source_cfg["revision"]}?blobs=true'
    metadata = json.loads(intake.small_get(api, 4*1024**2))
    entry = next(x for x in metadata['siblings'] if x['rfilename'] == source_cfg['archive'])
    if (metadata['sha'] != source_cfg['revision'] or metadata.get('gated') or metadata.get('disabled')
            or metadata.get('cardData', {}).get('license') != source_cfg['dataset_card_license']
            or entry['size'] != source_cfg['archive_bytes']
            or entry['lfs']['sha256'] != source_cfg['publisher_lfs_sha256_not_locally_verified']):
        raise ValueError('Changed pinned public source authority')
    references = []; fresh = 0
    with PublicRangeZip(reg['archive_url'], source_cfg['archive_bytes'],
            maximum_bytes=cfg['maximum_network_bytes_per_invocation'], block_size=cfg['range_block_bytes']) as stream:
        with zipfile.ZipFile(stream) as archive:
            if intake.inventory(archive) != read(ROOT/cfg['archive_inventory']):
                raise ValueError('Remote archive inventory changed')
            for number, row in enumerate(sorted(rows, key=lambda x: x['member']['offset']), 1):
                ref, new = cache_frame(row, reg, archive, cfg); references.append(ref); fresh += int(new)
                beat(state='past_image_cached', completed=number, planned=len(rows), fresh_images=fresh,
                    range_bytes=stream.transferred_bytes, range_requests=stream.range_requests,
                    node=row['node'], frame=row['annotation_frame'])
    return dict(images=references, fresh_images=fresh, range_bytes=stream.transferred_bytes,
        range_requests=stream.range_requests, remote_inventory_exact=True, publisher_identity_rechecked=True)


def raster(node, frame):
    path = PRIVATE/'rasters'/node/f'{frame:06d}.npz'; ref = read(path.with_suffix('.json'))
    with np.load(checked(ref['cache']), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pairs(cfg, reg, beat, pilot=False, replay=False):
    result = []
    for record in reg['roster']['records']:
        node = record['node']; frames = [record['query_frames'][0]] if pilot else range(1, max(record['query_frames'])+1)
        for frame in frames:
            value = method.estimate(raster(node, frame-1), raster(node, frame), cfg['measurement'])
            path = PRIVATE/'pairs'/node/f'{frame:06d}.json'
            output = stable(dict(node=node, frame=frame, registration_sha256=sha(REG), measurement=value))
            if replay:
                if read(path) != output:
                    raise ValueError('Past-image motion replay changed')
            else:
                once(path, output)
            result.append(dict(node=node, frame=frame, **value))
        beat(state='observed_pairs_complete', node=node, complete_pairs=len(result), replay=replay)
    return result


def summarize(rows, image_refs):
    centers = sum(r['native_centers'] for r in image_refs)
    inside = sum(r['native_centers_in_bounds'] for r in image_refs)
    result = []
    for node in sorted({r['node'] for r in rows}):
        selected = [r for r in rows if r['node'] == node]
        result.append(dict(node=node, pairs=len(selected), supported=sum(r['supported'] for r in selected),
            reasons={reason: sum(r['reason'] == reason for r in selected) for reason in sorted({r['reason'] for r in selected})}))
    return dict(pairs=len(rows), supported=sum(r['supported'] for r in rows),
        supported_fraction=sum(r['supported'] for r in rows)/len(rows), native_centers=centers,
        native_centers_in_bounds=inside, in_bounds_fraction=inside/centers if centers else None,
        by_recording=result, future_target_arrays_opened=False, predictive_lift_evaluated=False)


def pilot(cfg, reg, beat, resume):
    if (PUBLIC/'pilot.json').exists():
        raise ValueError('Completed pilot immutable; use replay instead')
    needs = {(r['node'], f) for r in reg['roster']['records'] for f in (r['query_frames'][0]-1, r['query_frames'][0])}
    acquired = acquire(cfg, reg, [r for r in reg['roster']['frames'] if (r['node'], r['annotation_frame']) in needs], beat, resume)
    measured = pairs(cfg, reg, beat, pilot=True)
    summary = summarize(measured, acquired['images'])
    passed = (summary['supported_fraction'] >= cfg['pilot_minimum_supported_fraction']
        and summary['in_bounds_fraction'] is not None and summary['in_bounds_fraction'] >= cfg['minimum_in_bounds_fraction'])
    once(PUBLIC/'pilot.json', stable(dict(result_source='fresh_run', summary=summary,
        measurement_engineering_screen_pass=passed, acquisition=acquired,
        new_predictive_admission=False, model_training=False, deployment_changed=False)))
    beat(state='pilot_complete', measurement_screen_pass=passed)


def full(cfg, reg, beat, resume):
    if not read(PUBLIC/'pilot.json')['measurement_engineering_screen_pass']:
        raise ValueError('Repair failed measurement pilot before full acquisition')
    if (PUBLIC/'measurement.json').exists():
        raise ValueError('Completed measurement immutable; use replay')
    acquired = acquire(cfg, reg, reg['roster']['frames'], beat, resume)
    measured = pairs(cfg, reg, beat)
    summary = summarize(measured, acquired['images'])
    if len(measured) != cfg['expected_pairs']:
        raise ValueError('Retain every predeclared past transition')
    passed = (summary['supported_fraction'] >= cfg['full_minimum_supported_fraction']
        and summary['in_bounds_fraction'] is not None and summary['in_bounds_fraction'] >= cfg['minimum_in_bounds_fraction'])
    once(PUBLIC/'measurement.json', stable(dict(result_source='fresh_run', summary=summary,
        measurement_engineering_screen_pass=passed, acquisition=acquired,
        new_predictive_admission=False, model_training=False, deployment_changed=False,
        original_rules_unchanged=True, stage5c_executed=False, smc_enabled=False)))
    beat(state='full_measurement_complete', measurement_screen_pass=passed)


def features(cfg, reg, beat, replay=False):
    if not read(PUBLIC/'measurement.json')['measurement_engineering_screen_pass']:
        raise ValueError('No predictive feature admission from failed measurement')
    data = past_arrays(cfg)
    out = np.empty((len(data['sites']), 48), np.float32); covered = np.zeros(len(out), bool)
    old = read(ROOT/'outputs/publication_readiness_2026_09/moving_producer_crossfit_v1/analysis.json')
    for item in old['forecasts']:
        with np.load(checked(item['forecast']), allow_pickle=False) as z:
            ix = z['local_ids']; prediction = z['relative_prediction']
            np.testing.assert_array_equal(z['global_ids'], data['global_ids'][ix])
        for record in reg['roster']['records']:
            at = data['recordings'][ix] == record['recording_index']
            for q in np.unique(data['frames'][ix][at]):
                local = np.flatnonzero(at&(data['frames'][ix] == q)); ids = ix[local]
                flows = {f: read(PRIVATE/'pairs'/record['node']/f'{f:06d}.json')['measurement'] for f in range(int(q)-6, int(q)+1)}
                out[ids] = method.contextual_features(data['history'][ids], prediction[local], int(q), flows)
                if covered[ids].any():
                    raise ValueError('Duplicate context rows')
                covered[ids] = True
        beat(state='causal_feature_bank_built', job=item['job']['key'], rows=int(covered.sum()))
    if not covered.all() or not np.isfinite(out).all():
        raise ValueError('Every original row requires explicit finite context')
    path = PRIVATE/'features.npy'; dest = PUBLIC/'feature_store.json'
    if replay:
        checked(read(dest)['features']); np.testing.assert_array_equal(np.load(path, allow_pickle=False), out)
    else:
        if path.exists():
            raise ValueError('Existing feature array needs replay,not overwrite')
        with path.with_suffix('.pending').open('wb') as f:
            np.save(f, out, allow_pickle=False)
        path.with_suffix('.pending').replace(path)
        once(dest, dict(result_source='fresh_run', rows=len(out), columns=48, features=artifact(path),
            global_ids=reg['past_arrays']['global_ids'], feature_names=list(method.NAMES),
            measurements_only_not_predictive_admission=True, future_target_arrays_opened=False,
            new_model_training=False, deployment_changed=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('register', 'pilot', 'full', 'features', 'replay'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, reg = register()
        if args.phase == 'register':
            print('Registered1851past images/1817transitions on125540fitting rows;no predictive admission.'); return
        committed(reg); start, beat = runtime(cfg, args.phase)
        before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase == 'pilot': pilot(cfg, reg, beat, args.resume)
        elif args.phase == 'full': full(cfg, reg, beat, args.resume)
        elif args.phase == 'features': features(cfg, reg, beat)
        else:
            result = read(PUBLIC/'measurement.json'); measured = pairs(cfg, reg, beat, replay=True)
            if summarize(measured, result['acquisition']['images']) != result['summary']:
                raise ValueError('Measurement scalar replay changed')
            features(cfg, reg, beat, replay=True)
            if before != {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}:
                raise ValueError('Replay changed public artifacts')
        dest = PRIVATE/'replays' if args.phase == 'replay' else PUBLIC/'executions'
        once(dest/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json', dict(phase=args.phase, pid=os.getpid(),
            seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            future_target_arrays_opened=False, neural_training=False))
        beat(state='phase_complete')


if __name__ == '__main__':
    main()
