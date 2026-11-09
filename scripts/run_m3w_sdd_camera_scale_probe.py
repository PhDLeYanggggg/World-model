"""Compare whole-image and tiled RAPiD on a fixed five-scene development subset."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_sdd_camera_support as study

cv2, np, torch = study.cv2, study.np, study.torch
from torchvision.ops import nms

PUBLIC, PRIVATE = study.PUBLIC / 'scale_probe', study.PRIVATE / 'scale_probe'
REG, FREEZE = PUBLIC / 'registration.json', PUBLIC / 'freeze.json'


def tiles(shape):
    h, w = shape[:2]
    th, tw = (int(np.ceil(x/1.8)) for x in (h, w))
    return [(x, y, tw, th) for y in (0, h-th) for x in (0, w-tw)]


def tiled(predict, image):
    boxes = []
    for x, y, w, h in tiles(image.shape):
        d = predict(image[y:y+h, x:x+w]).copy()
        d[:, :4] += [x, y, x, y]
        boxes.append(d)
    d = np.concatenate(boxes)
    keep = nms(torch.from_numpy(d[:, :4].copy()), torch.from_numpy(d[:, 4].copy()), .5)
    return d[keep.numpy()]


def freeze():
    reg, parent = study.checked_registration()
    if study.detector_contract() != reg['detector']:
        raise ValueError('Use the original RAPiD weights')
    full = study.PUBLIC / 'rapid_freeze.json'
    original = study.load(full)
    if original['registration'] != parent or original['label_geometry_used']:
        raise ValueError('Require the whole-image freeze')
    plan = {}
    for source in reg['sources']:
        if source['starts'] and source['scene'] not in plan:
            plan[source['scene']] = dict(source, selected_start=source['starts'][0])
    contract = dict(parent=parent, original=study.sha(full), code=study.sha(Path(__file__)),
                    selection='first registered window of first eligible clip in each scene',
                    tile_grid=[2, 2], adjacent_tile_overlap_fraction=.2, nms_iou=.5,
                    label_geometry_used=False, sources=plan)
    study.immutable(REG, contract)
    key = study.sha(REG)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    predict = study.rapid.rapid()
    timer, saved = time.monotonic(), {}
    for scene, src in plan.items():
        video, start = src['video'], src['selected_start']
        path = PRIVATE / f'{scene}.json'
        clip, _ = study.files(scene, video)
        if study.sha(clip) != src['clip_sha256']:
            raise ValueError('Registered camera clip changed')
        if not path.exists():
            cap = cv2.VideoCapture(str(clip))
            try:
                study.immutable(path, study.camera_window(cap, start, lambda image: tiled(predict, image), key, src['raster']))
            finally:
                cap.release()
        w = study.load(path)
        p = study.PRIVATE / 'rapid' / scene / video / f'{start:06}.json'
        prior = next(x for x in original['windows'] if (x['scene'], x['video'], x['start']) == (scene, video, start))
        if study.sha(p) != prior['sha256'] or w['registration'] != key:
            raise ValueError('Changed detector payload')
        a = [(f['frame'], f['pixel_sha256']) for f in w['frames']]
        b = [(f['frame'], f['pixel_sha256']) for f in study.load(p)['frames']]
        if a != b:
            raise ValueError('The resolution comparison needs identical frames')
        saved[scene] = dict(whole=study.sha(p), tiled=study.sha(path), video=video, start=start)
        beat = dict(pid=os.getpid(), scene=scene, done=len(saved), total=len(plan),
                    seconds=round(time.monotonic()-timer, 1), online=len(w['forecasts']))
        tmp = PRIVATE / 'heartbeat.pending'
        tmp.write_text(study.json.dumps(beat) + '\n')
        tmp.replace(PRIVATE / 'heartbeat.json')
        print(beat, flush=True)
    study.immutable(FREEZE, dict(registration=key, windows=saved, label_geometry_used=False))


def audit():
    reg, parent = study.checked_registration()
    contract, d = study.load(REG), study.load(FREEZE)
    if (contract['parent'] != parent or d['registration'] != study.sha(REG) or d['label_geometry_used']
            or contract['code'] != study.sha(Path(__file__)) or set(d['windows']) != set(contract['sources'])):
        raise ValueError('Require the frozen scale comparison')
    out = {}
    for scene, w in d['windows'].items():
        video, start = w['video'], w['start']
        _, ann = study.files(scene, video)
        src = contract['sources'][scene]
        if study.sha(ann) != src['annotation_sha256'] or (src['video'], src['selected_start']) != (video, start):
            raise ValueError('Annotation identity changed')
        labels = study.label_geometry(ann, set(range(start, start + study.OBS + study.FUTURE)))
        out[scene] = {}
        for name, p in (('whole', study.PRIVATE / 'rapid' / scene / video / f'{start:06}.json'),
                        ('tiled', PRIVATE / f'{scene}.json')):
            if study.sha(p) != w[name]:
                raise ValueError('Frozen scale outputs changed')
            counts, _, _ = study.paired_window(study.load(p), labels)
            out[scene][name] = dict(counts)
    study.immutable(PUBLIC / 'readout.json', dict(registration=study.sha(REG), freeze=study.sha(FREEZE),
                    source='fresh_run', results=out, diagnostic_only=True, independent_confirmation=False,
                    scale_only_with_cross_tile_suppression=True, no_model_fit=True))
    print({s: {k: (a['stable'], a.get('complete', 0)) for k, a in v.items()} for s, v in out.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'audit'))
    for p in (PUBLIC, PRIVATE):
        p.mkdir(parents=True, exist_ok=True)
    args = cli.parse_args()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'freeze': freeze, 'audit': audit}[args.phase]()
