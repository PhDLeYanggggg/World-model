"""Test whether a causal confidence bridge recovers SDD camera supervision."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_sdd_camera_raster as raster
from scripts import run_m3w_sdd_camera_aerial as aerial
from scripts import run_m3w_sdd_camera_support as study
from src.world_model.m3w_streaming_boxes import StreamingTracks, overlap

np, cv2, torch = study.np, study.cv2, study.torch
HOME = study.PRIVATE/'confidence_bridge_v2'
OUT = study.PUBLIC/'confidence_bridge_v2'
LOW, HIGH, ANCHORS, DISTINCT = .12, .30, 3, 4


def tracks(frames, minimum):
    tracker = StreamingTracks()
    records = []
    for j, f in enumerate(frames):
        box = np.asarray(f['boxes'], dtype=float).reshape(-1, 6)
        row, forecast = tracker.step(j, box[box[:, 4] >= minimum])
        records.append({k: f[k] for k in ('frame', 'pixel_sha256')} | dict(rows=row.tolist()))
    return dict(start=frames[0]['frame'], frames=records,
                forecasts=[[int(i), p.tolist(), o.tolist()] for i, p, o in forecast])


def histories(window):
    return {i: np.stack([next(r[1:] for r in f['rows'] if r[0] == i) for f in window['frames']])
            for i, *_ in window['forecasts']}


def bridge(high, low):
    strong, weak = histories(high), histories(low)
    selected = {}
    for i, boxes in weak.items():
        if (boxes[:, 4] >= HIGH).sum() < ANCHORS:
            continue
        # A low-threshold history may repair a dropout but may not replace an existing track.
        if any(sum(overlap(a[None, :4], b[None, :4])[0, 0] >= .3
                   for a, b in zip(boxes, z, strict=True)) >= DISTINCT for z in strong.values()):
            continue
        selected[i] = i + 1 + max(strong, default=-1)
    rows = []
    for h, l in zip(high['frames'], low['frames'], strict=True):
        rows.append(h | {'rows': [r for r in h['rows'] if r[0] in strong]
                         + [[selected[r[0]], *r[1:]] for r in l['rows'] if r[0] in selected]})
    return dict(start=high['start'], frames=rows, forecasts=high['forecasts'] +
                [[selected[i], p, o] for i, p, o in low['forecasts'] if i in selected])


def sources():
    d, reg = study.checked_registration()
    correction = raster.OUT/'registration.json'
    mapping = study.load(correction)
    previous = study.load(aerial.FREEZE)
    old_reg = study.load(aerial.REG)
    if (mapping['parent'] != reg or mapping['freezes']['aerial'] != study.sha(aerial.FREEZE)
            or previous['registration'] != study.sha(aerial.REG) or previous['label_geometry_used']
            or any(study.sha(study.ROOT/p) != h for p, h in old_reg['code'].items())):
        raise ValueError('Require the original audited raster and aerial detector')
    return d, mapping, {(w['scene'], w['video'], w['start']): w['sha256'] for w in previous['windows']}


def observe(cap, start, predict, old):
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        raise OSError('Cannot seek observation window')
    out = []
    for j, f in enumerate(old['frames']):
        t = start+j
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != t or t != f['frame']:
            raise ValueError('Source window frame index changed')
        ok, image = cap.read()
        if not ok or study.mobile.file_digest_array(image) != f['pixel_sha256']:
            raise ValueError('Camera inputs differ from the frozen baseline')
        out.append(dict(frame=t, pixel_sha256=f['pixel_sha256'], boxes=predict(image).tolist()))
    return out


def freeze():
    d, _, old = sources()
    reg = OUT/'registration.json'
    study.immutable(reg, dict(code_sha256=study.sha(Path(__file__)), raster=study.sha(raster.OUT/'registration.json'),
        aerial_freeze=study.sha(aerial.FREEZE), low_confidence=LOW, high_confidence=HIGH,
        bridge_min_high_frames=ANCHORS, bridge_duplicate_frames=DISTINCT, bridge_duplicate_iou=.3,
        role='same 92 existing SDD development windows only', labels_read=False))
    key = study.sha(reg)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    # This separate inference process changes only the confidence cut, not the frozen implementation.
    aerial.THRESHOLD = LOW
    one = aerial.predictor()
    predict = lambda im: aerial.scale.tiled(one, im)
    saved, start_time, replayed = [], time.monotonic(), False
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        clip, _ = study.files(scene, video)
        if study.sha(clip) != s['clip_sha256']:
            raise ValueError('Source camera clip changed')
        cap = cv2.VideoCapture(str(clip))
        try:
            for start in s['starts']:
                p = HOME/scene/video/f'{start:06}.json'
                p.parent.mkdir(parents=True, exist_ok=True)
                baseline = raster.read_window(old, 'aerial', scene, video, start)
                resumed = p.exists()
                if not resumed:
                    study.immutable(p, dict(registration=key, frames=observe(cap, start, predict, baseline)))
                payload = study.load(p)
                hi = tracks(payload['frames'], HIGH)
                if payload['registration'] != key or hi != {k: baseline[k] for k in hi}:
                    raise ValueError(f'High-confidence baseline did not reproduce: {scene}/{video}/{start}')
                if not replayed:
                    if payload['frames'] != observe(cap, start, predict, baseline):
                        raise ValueError('First window did not replay exactly')
                    replayed = True
                saved.append(dict(scene=scene, video=video, start=start, sha256=study.sha(p)))
                status = dict(pid=os.getpid(), completed=len(saved), total=len(old), resumed=resumed,
                              elapsed_seconds=round(time.monotonic()-start_time, 1))
                p_status = HOME/'heartbeat.pending'
                p_status.write_text(json.dumps(status)+'\n')
                p_status.replace(HOME/'heartbeat.json')
                if len(saved) % 4 == 0 or len(saved) == len(old):
                    print(status, flush=True)
        finally:
            cap.release()
    study.immutable(OUT/'freeze.json', dict(registration=key, windows=saved,
                    original_high_confidence_exact=True, first_window_exact=replayed, labels_read=False))


def audit():
    d, mapping, old = sources()
    reg = study.load(OUT/'registration.json')
    f = study.load(OUT/'freeze.json')
    if (f['registration'] != study.sha(OUT/'registration.json') or f['labels_read']
            or not f['original_high_confidence_exact'] or not f['first_window_exact']
            or reg['code_sha256'] != study.sha(Path(__file__))
            or reg['raster'] != study.sha(raster.OUT/'registration.json')):
        raise ValueError('Require a frozen label-blind confidence bank')
    entries = {(w['scene'], w['video'], w['start']): w['sha256'] for w in f['windows']}
    if entries.keys() != old.keys():
        raise ValueError('Require the exact original development roster')
    parent = study.load(raster.OUT/'readout.json')
    records, total, scenes = {}, defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        name = f'{scene}/{video}'
        _, ann = study.files(scene, video)
        if study.sha(ann) != s['annotation_sha256']:
            raise ValueError('Frozen annotation file changed')
        wanted = {t for st in s['starts'] for t in range(st, st+study.OBS+study.FUTURE)}
        labels = raster.convert(study.label_geometry(ann, wanted), mapping['mappings'][name]['reference_to_mp4_xy'])
        by_arm = defaultdict(Counter)
        for start in s['starts']:
            p = HOME/scene/video/f'{start:06}.json'
            if study.sha(p) != entries[(scene, video, start)]:
                raise ValueError('Frozen detector observations changed')
            frames = study.load(p)['frames']
            hi, lo = tracks(frames, HIGH), tracks(frames, LOW)
            b = bridge(hi, lo)
            for arm, w in dict(high=hi, low=lo, bridge=b).items():
                count, _, packed = study.paired_window(w, labels)
                count['wholly_unknown'] = count['online_histories'] - sum(bool(x[6].any()) for x in packed)
                by_arm[arm].update(count)
        baseline = parent['sources'][name]['aerial']
        if any(by_arm['high'][k] != baseline.get(k, 0) for k in ('online_histories', 'stable', 'complete')):
            raise ValueError('Corrected reference geometry did not reproduce')
        records[name] = {}
        for arm, c in by_arm.items():
            c['clips_with_support'] = int(c['complete'] >= 8 and c['windows_with_complete'] >= 3)
            total[arm].update(c)
            scenes[scene][arm].update(c)
            records[name][arm] = c
    study.immutable(OUT/'readout.json', dict(source='fresh_run', freeze_sha256=study.sha(OUT/'freeze.json'),
        original_raster_readout_sha256=study.sha(raster.OUT/'readout.json'), records=records, scenes=scenes,
        totals=total, selection_or_training_run=False, test_reopened=False, risk_threshold_changed=False))
    print({k: {m: v[m] for m in ('online_histories','complete','wholly_unknown','clips_with_support')}
           for k, v in total.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'audit'))
    a = parser.parse_args()
    for p in (HOME, OUT):
        p.mkdir(parents=True, exist_ok=True)
    with (HOME/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'freeze': freeze, 'audit': audit}[a.phase]()
