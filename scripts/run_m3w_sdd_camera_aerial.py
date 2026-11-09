"""Test an aerial-pretrained detector on the same frozen SDD development roster."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_m3w_sdd_camera_donors as donors
from scripts import run_m3w_sdd_camera_scale_probe as scale
from scripts import run_m3w_sdd_camera_support as study

cv2, np, torch, nms = study.cv2, study.np, study.torch, scale.nms
ARM = 'aerial'
PRIVATE, PUBLIC = study.PRIVATE / ARM, study.PUBLIC / ARM
REG, FREEZE = PUBLIC / 'registration.json', PUBLIC / 'freeze.json'
MODEL = 'visdrone-yolo26n-det-fp32-1344x768.onnx'
REV = 'ed4f6e79529666a452b3f89021a403a663e389d1'
SHA = '39596be0ae4b338eda72f6338d131d58e3cd4b32ceb0c25e14967f7a2d21aedb'
URL = f'https://huggingface.co/EdgeFirst/visdrone-yolo26-det/resolve/{REV}/1344x768/onnx/{MODEL}'
HEIGHT, WIDTH, THRESHOLD = 768, 1344, .3


def letterbox(image):
    h, w = image.shape[:2]
    ratio = min(WIDTH / w, HEIGHT / h)
    nw, nh = round(w * ratio), round(h * ratio)
    dx, dy = (WIDTH-nw)//2, (HEIGHT-nh)//2
    rgb = np.full((HEIGHT, WIDTH, 3), 114, dtype=np.uint8)
    rgb[dy:dy+nh, dx:dx+nw] = cv2.resize(image, (nw, nh))[:, :, ::-1]
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.
    return x, (dx, dy, nw/w, nh/h)


def decode(raw, transform, shape):
    h, w = shape[:2]
    dx, dy, sx, sy = transform
    d = raw[0].T
    scores, classes = d[:, 4:].max(1), d[:, 4:].argmax(1)
    allowed = (classes < 2) & (scores >= THRESHOLD)
    p = d[allowed, :4].astype('float64') * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
    scores = scores[allowed].astype('float64')
    lo, hi = p[:, :2] - .5*p[:, 2:], p[:, :2] + .5*p[:, 2:]
    p = np.column_stack((lo, hi))
    p = ((p - [dx, dy, dx, dy]) / [sx, sy, sx, sy]).clip(0, [w, h, w, h])
    valid = np.isfinite(p).all(1) & (p[:, 2:] > p[:, :2]).all(1)
    p, scores = p[valid], scores[valid]
    keep = nms(torch.from_numpy(p.copy()), torch.from_numpy(scores.copy()), .5).numpy()
    return np.column_stack((p[keep], scores[keep], np.ones(len(keep))))


def predictor():
    sys.path.insert(0, str(PRIVATE / 'runtime'))
    import onnxruntime as ort

    if ort.__version__ != '1.23.2' or study.sha(PRIVATE / MODEL) != SHA:
        raise ValueError('Require the pinned arm64 runtime and model export')
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    net = ort.InferenceSession(str(PRIVATE / MODEL), sess_options=options, providers=['CPUExecutionProvider'])
    meta = json.loads(net.get_modelmeta().custom_metadata_map['edgefirst'])
    if (net.get_inputs()[0].shape != [1, 3, HEIGHT, WIDTH]
            or net.get_outputs()[0].shape != [1, 14, 21168] or not meta['outputs'][0]['normalized']
            or meta['dataset']['classes'][:2] != ['pedestrian', 'people']
            or meta['validation']['preprocessing'] != 'letterbox' or meta['model']['end2end']):
        raise ValueError('Publisher ONNX output contract changed')

    def one(image):
        x, t = letterbox(image)
        raw, = net.run(None, {net.get_inputs()[0].name: x})
        return decode(raw, t, image.shape)

    return one


def freeze():
    d, parent = study.checked_registration()
    path = study.PUBLIC / 'rapid_freeze.json'
    ref = study.load(path)
    if ref['registration'] != parent or ref['label_geometry_used']:
        raise ValueError('Require the original matched camera freeze')
    before = {(w['scene'], w['video'], w['start']): w['sha256'] for w in ref['windows']}
    study.immutable(REG, dict(parent=parent, whole_rapid_freeze=study.sha(path), model_sha256=SHA, model_url=URL,
                              pretrained_on='VisDrone2019-DET as reported by model publisher',
                              confidence=THRESHOLD, nms_iou=.5, tile_grid=[2, 2], tile_overlap=.2,
                              workers=0, onnxruntime='1.23.2 CPU four intra-op threads',
                              code={str(Path(p).relative_to(study.ROOT)): study.sha(Path(p))
                                    for p in (__file__, scale.__file__, donors.__file__)},
                              role='development only', label_geometry_used=False))
    key = study.sha(REG)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    one = predictor()
    predict = lambda image: scale.tiled(one, image)
    timer, windows, cold = time.monotonic(), [], False
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        clip, _ = study.files(scene, video)
        if study.sha(clip) != s['clip_sha256']:
            raise ValueError('SDD clip changed')
        cap = cv2.VideoCapture(str(clip))
        try:
            for start in s['starts']:
                p = PRIVATE / scene / video / f'{start:06}.json'
                p.parent.mkdir(parents=True, exist_ok=True)
                resumed = p.exists()
                if not resumed:
                    study.immutable(p, study.camera_window(cap, start, predict, key, s['raster']))
                w = study.load(p)
                old = study.PRIVATE / 'rapid' / scene / video / p.name
                if (w['registration'] != key or study.sha(old) != before[(scene, video, start)]
                        or [(f['frame'], f['pixel_sha256']) for f in w['frames']] != [
                            (f['frame'], f['pixel_sha256']) for f in study.load(old)['frames']]):
                    raise ValueError('Compare the same decoded observations')
                if not cold:
                    if study.camera_window(cap, start, predict, key, s['raster']) != w:
                        raise ValueError('Aerial inference did not reproduce the first window')
                    cold = True
                windows.append(dict(scene=scene, video=video, start=start, sha256=study.sha(p)))
                msg = dict(pid=os.getpid(), phase=ARM, done=len(windows), total=len(before), scene=scene,
                           video=video, windows_resumed=resumed, track_histories=len(w['forecasts']),
                           seconds=round(time.monotonic()-timer, 1))
                tmp = PRIVATE / 'heartbeat.pending'
                tmp.write_text(json.dumps(msg) + '\n')
                tmp.replace(PRIVATE / 'heartbeat.json')
                print(json.dumps(msg), flush=True)
        finally:
            cap.release()
    study.immutable(FREEZE, dict(registration=key, windows=windows, repeated_first_window_exact=cold,
                                 label_geometry_used=False))


def audit():
    d, parent = study.checked_registration()
    key = study.sha(REG)
    contract, f = study.load(REG), study.load(FREEZE)
    if (contract['parent'] != parent or f['registration'] != key or f['label_geometry_used']
            or not f['repeated_first_window_exact']
            or any(study.sha(study.ROOT / p) != h for p, h in contract['code'].items())):
        raise ValueError('Require the matching frozen aerial bank')
    entries = {(w['scene'], w['video'], w['start']): w['sha256'] for w in f['windows']}
    recordings, scenes, total = {}, defaultdict(Counter), Counter()
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        _, ann = study.files(scene, video)
        if study.sha(ann) != s['annotation_sha256']:
            raise ValueError('Annotation lineage changed')
        labels = study.label_geometry(ann, {t for start in s['starts'] for t in range(start, start+20)})
        count, people, packed = Counter(), set(), []
        for start in s['starts']:
            path = PRIVATE / scene / video / f'{start:06}.json'
            if study.sha(path) != entries[(scene, video, start)]:
                raise ValueError('Aerial prefix changed')
            c, ids, rows = study.paired_window(study.load(path), labels)
            count.update(c)
            people.update(ids)
            packed.extend(rows)
        cache = study.PRIVATE / 'paired' / ARM / scene / f'{video}.npz'
        study.save_pairs(cache, packed)
        count['recordings'] = 1
        count['publisher_track_ids'] = len(people)
        count['recordings_with_support'] = int(count['complete'] >= 8 and count['windows_with_complete'] >= 3)
        data = dict(count, paired_cache_sha256=study.sha(cache))
        data['donor_export'] = donors.export(ARM, s, entries, data)
        recordings[f'{scene}/{video}'] = data
        scenes[scene].update(count)
        total.update(count)
    study.immutable(PUBLIC / 'readout.json', dict(source='fresh_run', registration=key,
        freeze=study.sha(FREEZE), recordings=recordings, scenes=scenes, total=total,
        independent_confirmation=False, fit_or_calibration_run=False, policy_changed=False))
    print(dict(total), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'audit'))
    args = cli.parse_args()
    for p in (PUBLIC, PRIVATE):
        p.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'freeze': freeze, 'audit': audit}[args.phase]()
