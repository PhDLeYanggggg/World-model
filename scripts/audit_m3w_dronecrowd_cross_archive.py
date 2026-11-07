"""Screen prospective external recordings against the used A/B image roster."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_dronecrowd_grouping import compare_clip_views

HOME = ROOT/'outputs/publication_readiness_2026_09'
DATA = ROOT/'data/stage_cvpr2027_experiments'
NAME = 'dronecrowd_cross_archive_v1'
PUBLIC, PRIVATE = HOME/NAME, DATA/NAME
OLD = HOME/'dronecrowd_grouping_v2'
MOVING = ('moving_past_background_v1', 'moving_past_background_own_v1')
CONFIG = ROOT/'configs/m3w_dronecrowd_grouping_v2.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_dronecrowd_cross_archive.py')


def read(p): return json.loads(p.read_text())
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
def once(p, value):
    if p.exists():
        if read(p) != value: raise ValueError(f'Changed receipt: {p.name}')
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_suffix('.pending'); temp.write_text(canonical(value)); temp.replace(p)


def select_views(frames):
    ordered = sorted(frames)
    if len(ordered) < 3: raise ValueError('Three distinct cached frames required')
    return [ordered[0], ordered[len(ordered)//2], ordered[-1]]


def moving_roster():
    records, roles = {}, {}
    for tag, name in zip(('A', 'B'), MOVING):
        registration = HOME/name/'registration.json'; reg = read(registration)
        for row in reg['roster']['frames']:
            node, frame = row['node'], row['annotation_frame']
            p = DATA/name/'rasters'/node/f'{frame:06d}.json'
            if not p.exists():
                # Exact A images are intentionally reused by the B raster roster.
                p = DATA/MOVING[0]/'rasters'/node/f'{frame:06d}.json'
            evidence = read(p)
            if evidence['row']['node'] != node or evidence['row']['annotation_frame'] != frame:
                raise ValueError('Rasters are not aligned to the frozen image rosters')
            ref = dict(frame=frame, cache=evidence['cache'], JPEG_sha256=evidence['JPEG_sha256'])
            frames = records.setdefault(node, {})
            if frame in frames and frames[frame]['JPEG_sha256'] != ref['JPEG_sha256']:
                raise ValueError('A/B shared image identity disagrees')
            frames.setdefault(frame, ref)
            roles.setdefault(node, set()).add(tag)
        if {r['node'] for r in reg['roster']['records']} != {n for n in roles if tag in roles[n]}:
            raise ValueError('Incomplete registered image recording roster')
    return [dict(node=node, roles=sorted(roles[node]), frames=[rows[f] for f in select_views(rows)])
            for node, rows in sorted(records.items())]


def external_roster(cfg):
    group = read(OLD/'exclusion_constraints.json')['recording_group']
    receipts = DATA/'dronecrowd_grouping_v2/features'
    identity = read(OLD/'run_identity.json')['run_id']
    result = []
    for node, g in sorted(group.items()):
        frames = []
        for f in cfg['selected_frames_one_based']:
            path = receipts/f'{node}_{f:03d}.npz'; ref = read(path.with_suffix('.json'))
            if ref['run_id'] != identity: raise ValueError('Changed source descriptor identity')
            frames.append(dict(frame=f, path=str(path.relative_to(ROOT)), sha256=ref['sha256']))
        result.append(dict(node=node, exclusion_group=g, frames=frames))
    return result


def register():
    cfg = read(CONFIG)
    parents = [CONFIG, OLD/'exclusion_constraints.json', OLD/'run_identity.json',
        ROOT/'src/evaluation/m3w_dronecrowd_grouping.py', *(HOME/p/'registration.json' for p in MOVING)]
    value = dict(files={str(p.relative_to(ROOT)): digest(p) for p in [*FILES, *parents]},
        exposed=moving_roster(), external=external_roster(cfg), settings=cfg, opencv=cv2.__version__,
        scope='source_image_screen_no_predictive_outcomes_or_role_assignment')
    once(PUBLIC/'registration.json', value)
    return value


def verified(ref):
    p = ROOT/ref['path']
    if digest(p) != ref['sha256']: raise ValueError('Changed image or descriptor artifact')
    return p


def cached_feature(ref):
    with np.load(verified(ref), allow_pickle=False) as z:
        d = z['descriptors']; return dict(xy=z['xy'], descriptors=d if len(d) else None)


def moving_feature(ref, cfg):
    with np.load(verified(ref['cache']), allow_pickle=False) as z:
        gray = cv2.resize(z['gray'], tuple(cfg['resize']), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(z['mask'], tuple(cfg['resize']), interpolation=cv2.INTER_NEAREST)
    gray = cv2.createCLAHE(clipLimit=cfg['clahe_clip_limit'], tileGridSize=(8, 8)).apply(gray)
    keypoints, d = cv2.SIFT_create(nfeatures=cfg['max_features']).detectAndCompute(gray, mask)
    if d is not None: d = np.sqrt(d/np.maximum(d.sum(1, keepdims=True), 1e-12))
    return dict(xy=np.array([p.pt for p in keypoints], np.float32).reshape(-1, 2), descriptors=d)


def quarantine(supported, group_map):
    touched = {group_map[x['external']] for x in supported}
    return sorted(touched), sorted(node for node, g in group_map.items() if g in touched)


def main():
    cli = argparse.ArgumentParser(); cli.add_argument('phase', choices=('register', 'screen', 'verify'))
    cli.add_argument('--max-new-recordings', type=int, default=0); args = cli.parse_args()
    cfg = register()
    if args.phase == 'register':
        print({k: len(cfg[k]) for k in ('exposed', 'external')}); return
    for p in (*FILES, PUBLIC/'registration.json'):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit frozen image screen before matching')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cv2.setNumThreads(1)
        start = time.monotonic(); refs, supported, n, tried, fresh = [], [], 0, 0, 0
        reg_sha = digest(PUBLIC/'registration.json'); matcher = cfg['settings']; keys = matcher['selected_frames_one_based']
        ext = {r['node']: {f['frame']: cached_feature(f) for f in r['frames']} for r in cfg['external']}
        for index, row in enumerate(cfg['exposed'], 1):
            if shutil.disk_usage(ROOT).free < 2*1024**3: raise OSError('Keep 2 GiB reserve; completed recordings remain resumable')
            if time.monotonic()-start > 43200: raise TimeoutError('Resume the same screen')
            p = PRIVATE/'records'/f'{index:03d}.json'
            if p.exists():
                v = read(p)
                for f in row['frames']: verified(f['cache'])
            else:
                if args.phase == 'verify': raise ValueError('Complete saved screen required')
                own = {k: moving_feature(f, matcher) for k, f in zip(keys, row['frames'])}
                comparisons = [dict(external=k, **compare_clip_views(own, e, matcher)) for k, e in ext.items()]
                v = dict(registration_sha256=reg_sha, exposed=row['node'], pairs=comparisons)
                once(p, v); fresh += 1
            if (v['registration_sha256'] != reg_sha or v['exposed'] != row['node']
                    or [r['external'] for r in v['pairs']] != list(ext)):
                raise ValueError('Incomplete or misaligned saved cross-archive results')
            refs.append(digest(p)); n += len(v['pairs']); tried += sum(x['frame_pairs_attempted'] for x in v['pairs'])
            supported.extend(dict(exposed=row['node'], **x) for x in v['pairs'] if x['status'] != 'unsupported')
            event = dict(pid=os.getpid(), phase=args.phase, records=index, total=len(cfg['exposed']),
                pairs=n, candidate_pairs=len(supported), fresh_records=fresh,
                seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            temp = PRIVATE/'heartbeat.pending'; temp.write_text(canonical(event)); temp.replace(PRIVATE/'heartbeat.json')
            print(event, flush=True)
            if args.max_new_recordings and fresh >= args.max_new_recordings and index < len(cfg['exposed']): return
        groups, clips = quarantine(supported, {r['node']:r['exclusion_group'] for r in cfg['external']})
        once(PUBLIC/'analysis.json', dict(registration_sha256=reg_sha, clip_pairs=n, frame_pairs=tried,
            exposed_recordings=len(cfg['exposed']), external_recordings=len(ext), supported_pairs=supported,
            quarantined_groups=groups, quarantined_clips=clips, record_receipt_sha256=refs,
            outcome_labels_opened=False, scientific_roles_assigned=False, independence_certified=False,
            scope='limited_to_frozen_A_B_image_roster_and_three_cached_views_per_recording',
            stage5c_executed=False, smc_enabled=False, submission_ready=False))
        print(dict(state='complete', fresh_records=fresh, analysis_sha256=digest(PUBLIC/'analysis.json')), flush=True)


if __name__ == '__main__':
    main()
