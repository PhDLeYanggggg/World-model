"""Qualify camera-derived histories on the intersection of the two SDD training splits."""

import argparse
from collections import Counter, defaultdict
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from scripts import run_m3w_waterloo_raw_prefix as mobile
from scripts import run_m3w_wepdtof_rapid_probe as rapid
from src.world_model.m3w_streaming_boxes import StreamingTracks, overlap

DATA = ROOT / 'external_data/StanfordDroneDataset'
SPLITS = ROOT / 'data/stage23_sdd_splits'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/sdd_camera_support_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/sdd_camera_support_v1'
REG = PUBLIC / 'registration.json'
OBS, FUTURE, WINDOWS = 8, 12, 4
ARMS = ('rapid', 'mobilenet')
sha, load, immutable = mobile.sha, rapid.load, mobile.immutable


def files(scene, video):
    return (DATA / 'video' / scene / video / 'video.mp4',
            DATA / 'annotations' / scene / video / 'annotations.txt')


def beat(**values):
    msg = dict(pid=os.getpid(), **values)
    path = PRIVATE / 'heartbeat.pending'
    path.write_text(json.dumps(msg) + '\n')
    path.replace(PRIVATE / 'heartbeat.json')
    print(json.dumps(msg), flush=True)


def visible_rows(path, frames=None):
    with path.open() as stream:
        for line in stream:
            cols = line.split()
            if len(cols) != 10:
                raise ValueError('Unexpected SDD annotation format')
            if cols[9] != '"Pedestrian"' or cols[6:8] != ['0', '0']:
                continue
            t = int(cols[5])
            if frames is None or t in frames:
                yield t, int(cols[0]), cols


def candidate_windows(visible, frame_count):
    possible = []
    length = OBS + FUTURE
    for s in range(0, frame_count-length+1, length):
        ids = set.intersection(*(visible.get(t, set()) for t in range(s, s+OBS)))
        if len(ids) >= 2:
            possible.append(s)
    if len(possible) <= WINDOWS:
        return possible
    return [possible[(2*k+1)*len(possible)//(2*WINDOWS)] for k in range(WINDOWS)]


def detector_contract():
    def git(*args):
        return subprocess.check_output(('git', '-C', str(rapid.CODE), *args), text=True).strip()

    r = load(rapid.REG)
    if (git('rev-parse', 'HEAD') != rapid.REV or git('status', '--porcelain', '--untracked-files=no')
            or r['upstream_commit'] != rapid.REV or sha(rapid.WEIGHTS) != r['weight_sha256']):
        raise ValueError('RAPiD checkout or published weights changed')
    m = sha(mobile.WEIGHTS)
    if not m.startswith('907ea3f9'):
        raise ValueError('MobileNet published weights changed')
    return dict(rapid=dict(checkpoint=r['weight_sha256'], upstream_commit=rapid.REV, confidence=.3,
                           short_side=1024, training=r['publisher_weight_training']),
                mobilenet=dict(checkpoint=m, confidence=.35, tile_grid=[4, 3], short_side=512,
                               training='Torchvision COCO weights'))


def train_intersection(paths):
    joint = []
    for p in paths:
        d = load(p)
        groups = {part: {(r['scene_id'], r['video_id']) for r in d[part]} for part in ('train', 'val', 'test')}
        if any(a & b for a, b in combinations(groups.values(), 2)):
            raise ValueError('Overlap in registered SDD split')
        joint.append(groups['train'])
    return sorted(set.intersection(*joint))


def register():
    splits = [SPLITS / f'{name}.json' for name in ('cross_scene_split', 'within_scene_video_split')]
    sources = []
    for scene, video in train_intersection(splits):
        clip, ann = files(scene, video)
        cap = cv2.VideoCapture(str(clip))
        try:
            if not cap.isOpened():
                raise OSError(f'Cannot open {scene}/{video}')
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            shape = [int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH)]
        finally:
            cap.release()
        people = defaultdict(set)
        for t, person, _ in visible_rows(ann):
            people[t].add(person)
        starts = candidate_windows(people, n)
        sources.append(dict(scene=scene, video=video, frame_count=n, raster=shape, starts=starts,
                            clip_sha256=sha(clip), annotation_sha256=sha(ann)))
    code = [Path(__file__), Path(mobile.__file__), Path(rapid.__file__),
            ROOT / 'src/world_model/m3w_streaming_boxes.py']
    immutable(REG, dict(role='development qualification only, both Stage23 train roles',
                        split_sha256={p.name: sha(p) for p in splits},
                        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
                        detector=detector_contract(), sources=sources, observed=OBS, future=FUTURE,
                        raw_frame_stride=1, windows_per_recording_max=WINDOWS,
                        roster='even mid-quantiles of disjoint 20-frame blocks with >=2 visible pedestrian IDs in all eight past frames; no future support filter',
                        association='per-frame Hungarian, IoU >= .10 and center distance <= half publisher box diagonal; retain identical pair in all eight past frames',
                        minimum_development_support='>=8 complete paired tracks across >=3 distinct windows per recording',
                        spatial_unit='pixel', seconds_verified=False, exposure='existing development data',
                        test_reopened=False, safety_protocol_changed=False, workers=0, torch_threads=4,
                        runtime=dict(architecture=platform.machine(), torch=str(torch.__version__), opencv=cv2.__version__)))
    print(f'Registered {len(sources)} clips and {sum(len(s["starts"]) for s in sources)} windows', flush=True)


def checked_registration():
    d = load(REG)
    if (any(sha(ROOT / p) != h for p, h in d['code_sha256'].items()) or
            any(sha(SPLITS / p) != h for p, h in d['split_sha256'].items())):
        raise ValueError('Registered split or implementation changed')
    return d, sha(REG)


def camera_window(cap, start, predict, reg, shape):
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        raise OSError(f'Cannot seek frame {start}')
    track, records = StreamingTracks(), []
    for j in range(OBS):
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start+j:
            raise OSError('MP4 frame position mismatch')
        ok, im = cap.read()
        if not ok or list(im.shape[:2]) != shape:
            raise OSError('Missing or differently sized past frame')
        rows, forecasts = track.step(j, predict(im))
        records.append(dict(frame=start+j, pixel_sha256=mobile.file_digest_array(im), rows=rows.tolist()))
    return dict(start=start, registration=reg, frames=records,
                forecasts=[[int(i), cv.tolist(), ols.tolist()] for i, cv, ols in forecasts])


def build(arm):
    d, reg = checked_registration()
    if detector_contract() != d['detector']:
        raise ValueError('Detector configuration changed')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    timer = time.monotonic()
    total = sum(len(s['starts']) for s in d['sources'])
    beat(phase=arm, done=0, total=total, seconds=0)
    if arm == 'rapid':
        predict = rapid.rapid()
    else:
        net = mobile.detector()

        def predict(im):
            boxes = mobile.detect(net, im)
            return boxes[boxes[:, -1] == 1]

    entries, cold = [], False
    for s in d['sources']:
        if not s['starts']:
            continue
        scene, video = s['scene'], s['video']
        clip, _ = files(scene, video)
        if sha(clip) != s['clip_sha256']:
            raise ValueError('Registered video changed')
        cap = cv2.VideoCapture(str(clip))
        try:
            if not cap.isOpened():
                raise OSError(f'Cannot open {scene}/{video}')
            for start in s['starts']:
                path = PRIVATE / arm / scene / video / f'{start:06}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                resumed = path.exists()
                if not resumed:
                    immutable(path, camera_window(cap, start, predict, reg, s['raster']))
                w = load(path)
                if w['registration'] != reg or [f['frame'] for f in w['frames']] != list(range(start, start+OBS)):
                    raise ValueError('Unbound per-window result')
                if not cold:
                    if camera_window(cap, start, predict, reg, s['raster']) != w:
                        raise ValueError('Repeated frame-only inference disagrees')
                    cold = True
                entries.append(dict(scene=scene, video=video, start=start, sha256=sha(path)))
                beat(phase=arm, done=len(entries), total=total, scene=scene, video=video,
                     windows_resumed=resumed, track_histories=len(w['forecasts']), seconds=round(time.monotonic()-timer, 1))
        finally:
            cap.release()
    immutable(PUBLIC / f'{arm}_freeze.json', dict(registration=reg, arm=arm, windows=entries,
                                                 repeated_first_window_exact=cold, label_geometry_used=False))


def associate(rows, labels):
    ids = sorted(labels)
    if not ids or not len(rows):
        return set()
    a, b = rows[:, 1:5], np.stack([labels[k][0] for k in ids])
    iou = overlap(a, b)
    center = lambda x: .5*(x[:, :2]+x[:, 2:])
    dist = np.linalg.norm(center(a)[:, None] - center(b)[None], axis=2)
    relative = dist / (.5*np.linalg.norm(b[:, 2:]-b[:, :2], axis=1)).clip(1.)
    valid = (iou >= .1) & (relative <= 1.)
    ii, jj = linear_sum_assignment(np.where(valid, 1-iou + .25*relative, 1e6))
    return {(int(rows[i, 0]), ids[j]) for i, j in zip(ii, jj, strict=True) if valid[i, j]}


def label_geometry(path, wanted):
    labels = defaultdict(dict)
    for t, person, cols in visible_rows(path, wanted):
        b = np.asarray(cols[1:5], dtype=np.float64)
        if (b[2:] <= b[:2]).any() or person in labels[t] or cols[8] not in ('0', '1'):
            raise ValueError('Invalid or duplicate SDD label geometry')
        labels[t][person] = (b, cols[8] == '0')
    return labels


def paired_window(w, labels):
    s = w['start']
    records = [np.asarray(f['rows'], dtype=float).reshape(-1, 7) for f in w['frames']]
    past = [labels.get(t, {}) for t in range(s, s+OBS)]
    matches = [associate(a, b) for a, b in zip(records, past, strict=True)]
    stable = set.intersection(*matches)
    bank = {k: (cv, ols) for k, cv, ols in w['forecasts']}
    if not {i for i, _ in stable} <= bank.keys():
        raise ValueError('An incomplete online history was matched as stable')
    count = Counter(windows=1, detection_boxes=sum(map(len, records)), label_boxes=sum(map(len, past)),
                    frame_matches=sum(map(len, matches)), online_histories=len(bank),
                    publisher_histories=len(set.intersection(*(set(f) for f in past))), stable=len(stable))
    ids, packed = set(), []
    for online, person in sorted(stable):
        hist = np.stack([a[a[:, 0] == online][0, 1:7] for a in records])
        future = [labels.get(t, {}).get(person) for t in range(s+OBS, s+OBS+FUTURE)]
        mask = np.array([z is not None for z in future])
        manual = np.array([z is not None and z[1] for z in future])
        y = np.stack([.5*(z[0][:2]+z[0][2:]) if z else np.zeros(2) for z in future])
        count['complete'] += int(mask.all())
        count['no_future'] += int(not mask.any())
        count['valid_future_steps'] += int(mask.sum())
        count['non_generated_future_steps'] += int(manual.sum())
        count['all_future_steps_non_generated'] += int(manual.all())
        ids.add(person)
        packed.append((s, online, person, hist, np.asarray(bank[online]), y, mask, manual))
    count['windows_with_complete'] = int(count['complete'] > 0)
    count['windows_with_two_complete'] = int(count['complete'] >= 2)
    return count, ids, packed


def save_pairs(path, packed):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.npz')
    schema = (('start', (), 'int64'), ('online_id', (), 'int64'), ('publisher_id', (), 'int64'),
              ('past_xyxy_conf_class', (OBS, 6), 'float64'), ('cv_ols4', (2, FUTURE, 2), 'float64'),
              ('target_xy', (FUTURE, 2), 'float64'), ('target_mask', (FUTURE,), 'bool'),
              ('non_generated_mask', (FUTURE,), 'bool'))
    np.savez(tmp, **{key: np.asarray([r[i] for r in packed], dtype=dtype).reshape((-1, *shape))
                     for i, (key, shape, dtype) in enumerate(schema)})
    tmp.replace(path)


def audit():
    d, reg = checked_registration()
    refs = {a: PUBLIC / f'{a}_freeze.json' for a in ARMS}
    frozen = {a: load(p) for a, p in refs.items()}
    expected = [(s['scene'], s['video'], t) for s in d['sources'] for t in s['starts']]
    bound = {}
    for a, f in frozen.items():
        if (f['registration'] != reg or f['label_geometry_used'] or not f['repeated_first_window_exact']
                or [(w['scene'], w['video'], w['start']) for w in f['windows']] != expected):
            raise ValueError('Complete, verified, matched freezes are required before geometry audit')
        bound[a] = {k: w['sha256'] for k, w in zip(expected, f['windows'], strict=True)}
    sources, totals, scenes = {}, defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        _, ann = files(scene, video)
        if sha(ann) != s['annotation_sha256']:
            raise ValueError('Registered annotations changed')
        wanted = {t for start in s['starts'] for t in range(start, start+OBS+FUTURE)}
        labels = label_geometry(ann, wanted) if wanted else {}
        counts, ids, packs = (defaultdict(factory) for factory in (Counter, set, list))
        for start in s['starts']:
            pixels = None
            for a in ARMS:
                path = PRIVATE / a / scene / video / f'{start:06}.json'
                if sha(path) != bound[a][scene, video, start]:
                    raise ValueError('Frozen camera window changed')
                w = load(path)
                p = [f['pixel_sha256'] for f in w['frames']]
                if pixels is not None and p != pixels:
                    raise ValueError('Arms did not see identical decoded frames')
                pixels = p
                count, people, packed = paired_window(w, labels)
                counts[a].update(count)
                ids[a].update(people)
                packs[a].extend(packed)
        src = sources[f'{scene}/{video}'] = {}
        for a in ARMS:
            path = PRIVATE / 'paired' / a / scene / f'{video}.npz'
            save_pairs(path, packs[a])
            counts[a]['recordings'] = 1
            counts[a]['recordings_with_support'] = int(counts[a]['complete'] >= 8 and counts[a]['windows_with_complete'] >= 3)
            counts[a]['publisher_track_ids'] = len(ids[a])
            totals[a].update(counts[a])
            scenes[scene][a].update(counts[a])
            src[a] = dict(counts[a], paired_cache_sha256=sha(path))
    immutable(PUBLIC / 'readout.json', dict(registration=reg, freezes={a: sha(p) for a, p in refs.items()},
                                           source='fresh_run', recordings=sources, scenes=scenes, totals=totals,
                                           train_or_calibration_run=False, independent_confirmation=False,
                                           test_reopened=False, policy_changed=False, safety_protocol_changed=False))
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', *ARMS, 'audit'))
    args = parser.parse_args()
    for p in (PRIVATE, PUBLIC):
        p.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase in ARMS:
            build(args.phase)
        else:
            {'register': register, 'audit': audit}[args.phase]()
