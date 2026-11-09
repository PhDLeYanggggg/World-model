"""Check camera-to-history support on AerialMPT publisher-training clips."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_sdd_camera_aerial as aerial
from scripts import run_m3w_sdd_camera_confidence as confidence
from scripts import run_m3w_sdd_camera_support as study

np, torch, cv2 = study.np, study.torch, study.cv2
TAG = 'aerialmpt_camera_prefix_v1'
ROOT = study.ROOT
OUT = study.PUBLIC.parent/TAG
PRIVATE = study.PRIVATE.parent/TAG
SOURCE = ROOT/'external_data/AerialMPT_source_audit/publisher.zip'
INTAKE = OUT.parent/'topdown_source_intake_v1/aerialmpt.json'
OBS, FUTURE = study.OBS, study.FUTURE
RADII = (2., 4., 8.)


def registration(archive):
    prior = study.load(INTAKE)['aerial']
    if study.sha(SOURCE) != prior['asset']['sha256']:
        raise ValueError('Publisher archive does not match the source audit')
    names = set(archive.namelist())
    windows, short = {}, {}
    for key in sorted(names):
        m = re.fullmatch(r'train/([A-Za-z0-9_]+)/image_list\.txt', key)
        if m is None:
            continue
        clip = m[1]
        ids = archive.read(key).decode().split()
        if len(ids) != len(set(ids)) or len(ids) != prior['clips'][clip]['frames']:
            raise ValueError('Released image grid changed')
        images = [f'train/{clip}/{i}.png' for i in ids]
        if not set(images) <= names:
            raise ValueError('Missing original image member')
        if len(ids) >= OBS+FUTURE:
            windows[clip] = dict(frames=len(ids), observation_members=images[:OBS])
        else:
            short[clip] = len(ids)
    if len(windows)+len(short) != 8:
        raise ValueError('Expected the eight released training clips')
    reg = OUT/'registration.json'
    base = study.load(aerial.REG)
    if base['model_sha256'] != aerial.SHA or any(study.sha(ROOT/p) != s for p, s in base['code'].items()):
        raise ValueError('Aerial detector implementation changed')
    study.immutable(reg, dict(archive=prior['asset'], selection='first complete 20-image grid of each publisher train clip',
        code=study.sha(Path(__file__)), confidence_code=study.sha(Path(confidence.__file__)),
        sdd_contract=study.sha(study.REG), detector=study.sha(aerial.REG),
        detector_min_score=.12, score_thresholds=[.30, .12], grids=[2, 4], nms_iou=.5,
        match_radius_px=4., sensitivity_radius_px=[2., 8.], windows=windows, too_short=short,
        observation_steps=OBS, prediction_steps=FUTURE, label_rows_read=False))
    return study.load(reg), study.sha(reg)


def dense_tiles(length, n):
    size = int(np.ceil(length/(1+.8*(n-1))))
    return [(int(lo), size) for lo in np.rint(np.linspace(0, length-size, n))]


def dense_predict(one, image):
    d = []
    for y, h in dense_tiles(image.shape[0], 4):
        for x, w in dense_tiles(image.shape[1], 4):
            box = one(image[y:y+h, x:x+w]).copy()
            box[:, :4] += [x, y, x, y]
            d.append(box)
    boxes = np.concatenate(d)
    keep = aerial.nms(torch.from_numpy(boxes[:, :4].copy()), torch.from_numpy(boxes[:, 4].copy()), .5)
    return boxes[keep.numpy()]


def observe(archive, members, predictors):
    frames = {name: [] for name in predictors}
    for j, name in enumerate(members):
        im = cv2.imdecode(np.frombuffer(archive.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.ndim != 3:
            raise ValueError('Cannot decode released observation image')
        stamp = study.mobile.file_digest_array(im)
        for arm, predict in predictors.items():
            frames[arm].append(dict(frame=j, pixel_sha256=stamp, boxes=predict(im).tolist()))
    return frames


def freeze(archive, reg, key):
    study.checked_registration()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    aerial.THRESHOLD = reg['detector_min_score']
    one = aerial.predictor()
    predictors = {'grid2': lambda im: aerial.scale.tiled(one, im),
                  'grid4': lambda im: dense_predict(one, im)}
    saved, replayed, timer = {}, False, time.monotonic()
    for clip, w in reg['windows'].items():
        path = PRIVATE/f'{clip}.json'
        resumed = path.exists()
        if not resumed:
            study.immutable(path, dict(registration=key, frames=observe(archive, w['observation_members'], predictors)))
        record = study.load(path)
        if record['registration'] != key:
            raise ValueError('Observation bank is not bound to this protocol')
        if not replayed:
            if record['frames'] != observe(archive, w['observation_members'], predictors):
                raise ValueError('First observation bank did not replay exactly')
            replayed = True
        saved[clip] = study.sha(path)
        status = dict(pid=os.getpid(), done=len(saved), total=len(reg['windows']), clip=clip,
                      seconds=round(time.monotonic()-timer, 1), resumed=resumed)
        tmp = PRIVATE/'heartbeat.pending'
        tmp.write_text(json.dumps(status)+'\n')
        tmp.replace(PRIVATE/'heartbeat.json')
        print(status, flush=True)
    study.immutable(OUT/'freeze.json', dict(registration=key, recordings=saved, first_replay_exact=replayed,
                                            label_rows_read=False))


def labels(archive, clip, frame_count):
    out = defaultdict(dict)
    with archive.open(f'train/{clip}/{clip}_gts.txt') as fh:
        for line in fh:
            frame, agent, x, y, w, h, flag, a, b = line.decode().split()
            t, k = int(frame)-1, int(agent)
            p = np.asarray([float(x)+2, float(y)+2])
            if not (0 <= t < frame_count and k >= 0 and (w, h, flag, a, b) == ('4', '4', '1', '-1', '-1')
                    and np.isfinite(p).all() and k not in out[t]):
                raise ValueError('Unexpected point annotation encoding')
            if t < OBS+FUTURE:
                out[t][k] = p
    return out


def associate(rows, annotation, radius):
    ids = sorted(annotation)
    if not len(rows) or not ids:
        return set()
    xy = .5*(rows[:, 1:3]+rows[:, 3:5])
    d = np.linalg.norm(xy[:, None, :] - np.stack([annotation[i] for i in ids])[None], axis=-1)
    i, j = study.linear_sum_assignment(np.where(d <= radius, d, 1e9))
    return {(int(rows[k, 0]), ids[l]) for k, l in zip(i, j, strict=True) if d[k, l] <= radius}


def assess(window, target, radius):
    observed = [target.get(t, {}) for t in range(OBS)]
    rows = [np.asarray(f['rows'], dtype=float).reshape(-1, 7) for f in window['frames']]
    assignments = [associate(a, b, radius) for a, b in zip(rows, observed, strict=True)]
    stable = dict(set.intersection(*assignments))
    same_target = set.intersection(*({k for _, k in match} for match in assignments))
    references = set.intersection(*(set(t) for t in observed))
    future = [target.get(t, {}) for t in range(OBS, OBS+FUTURE)]
    expected = set.intersection(*(set(t) for t in future))
    forecasts = {k: np.asarray([cv, ols]) for k, cv, ols in window['forecasts']}
    if not stable.keys() <= forecasts.keys():
        raise ValueError('Stable matched identities must have a causal forecast')
    errs = [np.linalg.norm(forecasts[k] - np.stack([t[p] for t in future]), axis=-1)
            for k, p in stable.items() if p in expected]
    complete = len(errs)
    return dict(label_points=sum(map(len, observed)), matched_points=sum(map(len, assignments)),
        detection_points=sum(map(len, rows)), reference_histories=len(references),
        reference_complete=len(references & expected), eight_points_ignoring_online_id=len(same_target),
        online=len(forecasts), stable=len(stable), complete=complete,
        wholly_unknown=len(forecasts)-sum(any(p in t for t in future) for p in stable.values()),
        fully_labeled_joint=complete >= 2 and complete == len(forecasts),
        complete_ade_px=np.mean(errs, axis=(0, 2)).tolist() if errs else None,
        complete_fde_px=np.mean(np.stack(errs)[:, :, -1], axis=0).tolist() if errs else None)


def score(archive, reg, key):
    freeze_path = OUT/'freeze.json'
    f = study.load(freeze_path)
    if (f['registration'] != key or f['label_rows_read'] or not f['first_replay_exact']
            or f['recordings'].keys() != reg['windows'].keys()):
        raise ValueError('Require the exact observation freeze before reading labels')
    totals = defaultdict(lambda: defaultdict(Counter))
    results = {}
    for clip, w in reg['windows'].items():
        path = PRIVATE/f'{clip}.json'
        if study.sha(path) != f['recordings'][clip]:
            raise ValueError('Frozen detector bank changed')
        target = labels(archive, clip, w['frames'])
        results[clip] = {}
        for grid, frames in study.load(path)['frames'].items():
            hi, lo = (confidence.tracks(frames, t) for t in (.30, .12))
            for key0, window in dict(high=hi, low=lo, bridge=confidence.bridge(hi, lo)).items():
                arm = f'{grid}_{key0}'
                by_radius = {str(r): assess(window, target, r) for r in RADII}
                results[clip][arm] = by_radius
                for rad, v in by_radius.items():
                    totals[arm][rad].update({k: int(x) for k, x in v.items() if x is not None and not isinstance(x, list)})
    study.immutable(OUT/'readout.json', dict(source='fresh_run', freeze=study.sha(freeze_path),
        totals=totals, clips=results, publisher_partition='train', test_members_read=False,
        unit='video pixels and released image steps', main_match_radius_px=4.,
        role='camera-input development diagnostic; no policy calibration, no independent confirmation'))
    for arm, r in totals.items():
        a = r['4.0']
        print(arm, {k: a[k] for k in ('matched_points', 'label_points', 'eight_points_ignoring_online_id',
                                     'online', 'stable', 'complete', 'fully_labeled_joint')})


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'score'))
    args = cli.parse_args()
    for p in (PRIVATE, OUT):
        p.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock, ZipFile(SOURCE) as archive:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        setup, identity = registration(archive)
        (freeze if args.phase == 'freeze' else score)(archive, setup, identity)
