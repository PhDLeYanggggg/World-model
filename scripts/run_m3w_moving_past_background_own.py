"""Same image measurement on the complete,already-exposed own-B inputs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_past_background as base

np, cv2 = base.np, base.cv2
read, sha, once, stable, artifact, checked = base.read, base.sha, base.once, base.stable, base.artifact, base.checked
PARENT_PUBLIC, PARENT_PRIVATE, PARENT_REG = base.PUBLIC, base.PRIVATE, base.REG
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_past_background_own_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_past_background_own_v1'
CONFIG = ROOT/'configs/m3w_moving_past_background_own_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_past_background_own.py')
original_cache, original_raster = base.cache_frame, base.raster
_parent_frames = None


def past_members(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in base.PAST_KEYS}


def plan(cfg):
    manifest = read(ROOT/cfg['parent_population']); data = past_members(checked(manifest['cache']))
    admission = read(ROOT/cfg['admission']); nodes = sorted(admission['recordings'])
    role = read(ROOT/cfg['role_registration']); allowed = {s for n in role['jobs'] for s in n['part']['names']}
    if len(data['sites']) != 62466 or set(data['sites']) != allowed or len(allowed) != 8:
        raise ValueError('Complete old own-B roster required')
    inventory = read(ROOT/cfg['archive_inventory']); images = {}
    for item in inventory:
        key = base.clip_key(item['name'], 'frames')
        if key is not None:
            images[key, int(Path(item['name']).stem)] = item
    parent = read(PARENT_REG); parent_frames = {(r['node'], r['annotation_frame']): r for r in parent['roster']['frames']}
    records, frames = [], []
    for index in sorted(set(data['recordings'])):
        node = nodes[int(index)]; meta = admission['recordings'][node]; at = data['recordings'] == index
        queries = np.unique(data['frames'][at]); body = ROOT/meta['source_annotation_path']
        if (queries.min() < 7 or set(data['sites'][at]) != {meta['exclusion_group']}
                or sha(body) != meta['source_annotation_sha256']):
            raise ValueError('Changed causal image/annotation/source bridge')
        records.append(dict(node=node, recording_index=int(index), source=meta['exclusion_group'],
            clip=meta['clip'], query_frames=queries.tolist(), rows=int(at.sum()), annotation=artifact(body)))
        for frame in range(int(queries.max())+1):
            row = dict(node=node, annotation_frame=frame, image_frame=frame+1, member=images[meta['clip'], frame+1])
            if (node, frame) in parent_frames and row != parent_frames[node, frame]:
                raise ValueError('Parent raster source member differs')
            frames.append(row)
    fresh = [r for r in frames if (r['node'], r['annotation_frame']) not in parent_frames]
    if (len(records) != 24 or len(frames) != 1255 or len(frames)-len(records) != 1231
            or sum(len(r['query_frames']) for r in records) != 1087 or len(fresh) != 996
            or sum(r['member']['compressed_bytes'] for r in fresh) != 522029182):
        raise ValueError('No reduced or outcome-selected image cohort allowed')
    return dict(records=records, frames=frames, rows=62466, queries=1087, image_members=1255,
        transitions=1231, fresh_images=996, exact_parent_images=259, future_target_arrays_opened=False)


def register():
    cfg = read(CONFIG); previous = read(PARENT_REG); parent = read(ROOT/cfg['parent_measurement'])
    if (not parent['measurement_engineering_screen_pass'] or cfg['measurement'] != previous['config']['measurement']
            or not cfg['measurement_only'] or not cfg['original_rules_unchanged']
            or any(cfg[k] for k in ('retain_raw_JPEG', 'future_target_arrays_opened', 'new_predictive_admission',
                'new_outer_C_H_readout', 'identity_certification', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Same measurement and original rules,not predictive admission')
    roster = plan(cfg); source = base.intake.cached_source()
    result = stable(dict(config=cfg, roster=roster, files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in (Path(base.__file__), Path(base.method.__file__))},
        parents=[artifact(ROOT/cfg[k]) for k in ('parent_measurement', 'parent_population', 'role_registration', 'admission')],
        parent_image_registration=artifact(PARENT_REG), archive_url=source['archive_url'],
        source_config=read(ROOT/cfg['source_config']), opencv=cv2.__version__, numpy=np.__version__,
        role='already_exposed_own_B_past_input_measurement_not_new_confirmation',
        original_rules_unchanged=True, future_target_arrays_opened=False, new_predictive_admission=False))
    once(REG, result); return cfg, result


def committed(reg):
    for p in (*FILES, REG):
        blob = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != sha(p):
            raise ValueError('Commit exact own-B input measurement before acquisition')
    for path, digest in reg['dependencies'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Changed parent image measurement')


def parent_frames():
    global _parent_frames
    if _parent_frames is None:
        _parent_frames = {(r['node'], r['annotation_frame']): r for r in read(PARENT_REG)['roster']['frames']}
    return _parent_frames


def cache_frame(row, reg, archive, cfg):
    key = row['node'], row['annotation_frame']
    if key not in parent_frames():
        return original_cache(row, reg, archive, cfg)
    path = PARENT_PRIVATE/'rasters'/row['node']/f'{row["annotation_frame"]:06d}.json'
    meta = read(path); checked(meta['cache'])
    if meta['row'] != row or meta['registration_sha256'] != sha(PARENT_REG):
        raise ValueError('Changed exact parent-image reuse')
    return meta, False


def raster(node, frame):
    if (node, frame) not in parent_frames():
        return original_raster(node, frame)
    path = PARENT_PRIVATE/'rasters'/node/f'{frame:06d}.npz'; meta = read(path.with_suffix('.json'))
    with np.load(checked(meta['cache']), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('register', 'measure', 'replay'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, reg = register()
        if args.phase == 'register':
            print('Registered all62466own-B past inputs,996new images/259exact reused;no target readout.'); return
        committed(reg)
        # Process-local adapter; original A definitions and caches are never edited.
        base.PUBLIC, base.PRIVATE, base.REG, base.CONFIG = PUBLIC, PRIVATE, REG, CONFIG
        base.cache_frame, base.raster = cache_frame, raster
        start, beat = base.runtime(cfg, args.phase)
        before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase == 'measure':
            if (PUBLIC/'measurement.json').exists():
                raise ValueError('Completed own-B measurement immutable; use replay')
            acquired = base.acquire(cfg, reg, reg['roster']['frames'], beat, args.resume)
            measured = base.pairs(cfg, reg, beat); summary = base.summarize(measured, acquired['images'])
            passed = summary['supported_fraction'] >= .8 and summary['in_bounds_fraction'] >= .99
            once(PUBLIC/'measurement.json', stable(dict(result_source='fresh_run_with259cached_verified_rasters',
                summary=summary, acquisition=acquired, measurement_engineering_screen_pass=passed,
                original_rules_unchanged=True, future_target_arrays_opened=False, new_predictive_admission=False)))
        else:
            old = read(PUBLIC/'measurement.json'); measured = base.pairs(cfg, reg, beat, replay=True)
            if base.summarize(measured, old['acquisition']['images']) != old['summary']:
                raise ValueError('Own-B measurement replay differs')
            if before != {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}:
                raise ValueError('Replay changed public artifacts')
        dest = PRIVATE/'replays' if args.phase == 'replay' else PUBLIC/'executions'
        once(dest/f'{args.phase}_{time.time_ns()}_{base.os.getpid()}.json', dict(pid=base.os.getpid(), phase=args.phase,
            seconds=time.monotonic()-start, peak_RSS_bytes=base.resource.getrusage(base.resource.RUSAGE_SELF).ru_maxrss,
            future_target_arrays_opened=False, neural_training=False))
        beat(state='phase_complete')


if __name__ == '__main__':
    main()
