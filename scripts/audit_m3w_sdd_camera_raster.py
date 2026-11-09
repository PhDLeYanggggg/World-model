"""Repair annotation-to-MP4 raster units without changing any camera predictions."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_sdd_camera_coverage as coverage
from scripts import audit_m3w_sdd_camera_motion as motion
from scripts import run_m3w_sdd_camera_aerial as aerial
from scripts import run_m3w_sdd_camera_scale_probe as scale_probe
from scripts import run_m3w_sdd_camera_support as study

np, cv2 = study.np, study.cv2
OUT = study.PUBLIC / 'raster_registration'


def convert(labels, scale):
    return {t: {i: (box*np.tile(scale, 2), origin) for i, (box, origin) in rows.items()}
            for t, rows in labels.items()}


def outside(labels, shape):
    h, w = shape
    return sum(bool((center < 0).any() or (center >= [w, h]).any()) for f in labels.values()
               for b, _ in f.values() for center in [(b[:2]+b[2:])/2.])


def landmark_receipt(s, cached):
    scene, video = s['scene'], s['video']
    clip, ann = study.files(scene, video)
    ref = ann.parent / 'reference.jpg'
    image = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
    if image is None or not cached:
        raise ValueError('Need publisher reference and registered past windows')
    h, w = s['raster']
    oh, ow = image.shape
    sift, matcher = cv2.SIFT_create(nfeatures=1300), cv2.BFMatcher(cv2.NORM_L2)
    k, d = sift.detectAndCompute(cv2.resize(image, (w, h)), None)
    cap = cv2.VideoCapture(str(clip))
    rows = []
    try:
        for frame in (cached[0]['frames'][0], cached[-1]['frames'][0]):
            t = frame['frame']
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, t):
                raise OSError('Cannot seek a registered past frame')
            ok, im = cap.read()
            if not ok or study.mobile.file_digest_array(im) != frame['pixel_sha256']:
                raise ValueError('Registration views differ from frozen detector input')
            p, z = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
            good = [m for pair in matcher.knnMatch(d, z, k=2) if len(pair) == 2
                    for m, n in [pair] if m.distance < .75*n.distance]
            x = np.asarray([k[m.queryIdx].pt for m in good])
            y = np.asarray([p[m.trainIdx].pt for m in good])
            if len(good) < 30:
                raise ValueError(f'Insufficient static landmarks: {scene}/{video} {t}')
            mat, mask = cv2.estimateAffinePartial2D(x, y, ransacReprojThreshold=3.)
            mask = mask[:, 0].astype(bool)
            err = np.linalg.norm(y - x @ mat[:, :2].T - mat[:, 2], axis=1)
            identity_err = np.linalg.norm(y-x, axis=1)
            rows.append(dict(frame=t, matches=len(x), inliers=int(mask.sum()),
                             identity_median_px=float(np.median(identity_err)),
                             affine_inlier_median_px=float(np.median(err[mask])),
                             fitted_affine_after_resize=mat.tolist()))
    finally:
        cap.release()
    if rows[0]['inliers'] < 50 or rows[0]['identity_median_px'] > 4. or abs((w/ow)/(h/oh)-1) > .03:
        raise ValueError(f'Simple raster resize is not supported: {scene}/{video}')
    return dict(reference_sha256=study.sha(ref), reference_hw=[oh, ow], mp4_hw=[h, w],
                reference_to_mp4_xy=[w/ow, h/oh], landmark_checks=rows)


def read_window(lookup, arm, scene, video, start):
    path = study.PRIVATE / arm / scene / video / f'{start:06}.json'
    if study.sha(path) != lookup[(scene, video, start)]:
        raise ValueError('Frozen detector cache changed')
    return study.load(path)


def run():
    d, reg = study.checked_registration()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {a: study.PUBLIC / f'{a}_freeze.json' for a in study.ARMS} | {aerial.ARM: aerial.FREEZE}
    parent = study.load(study.PUBLIC / 'readout.json')
    extra = study.load(aerial.PUBLIC / 'readout.json')
    expected = dict(parent['freezes'], aerial=extra['freeze'])
    if parent['registration'] != reg or any(study.sha(p) != expected[a] for a, p in refs.items()):
        raise ValueError('Require the original frozen prediction banks')
    lookups = {a: {(r['scene'], r['video'], r['start']): r['sha256'] for r in study.load(p)['windows']}
               for a, p in refs.items()}
    # Establish the image raster relation before loading any trajectory geometry.
    mappings = {}
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        w = [read_window(lookups['rapid'], 'rapid', scene, video, t) for t in s['starts']]
        mappings[f'{scene}/{video}'] = landmark_receipt(s, w)
    receipt = OUT / 'registration.json'
    study.immutable(receipt, dict(parent=reg, freezes=expected, mappings=mappings,
        mapping='publisher reference raster size to video raster size; no per-frame warp or scale fitting to trajectories',
        purpose='map evaluation labels only, never model input',
        implementation_sha256=study.sha(Path(__file__)), seconds_or_metric_calibration=False))

    totals, by_scene = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    errors, geometry, sources, scale_labels = defaultdict(list), {}, {}, {}
    probe = study.load(scale_probe.FREEZE)
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        name = f'{scene}/{video}'
        _, ann = study.files(scene, video)
        if study.sha(ann) != s['annotation_sha256']:
            raise ValueError('Registered labels changed')
        wanted = {t for st in s['starts'] for t in range(st, st+study.OBS+study.FUTURE)}
        raw = study.label_geometry(ann, wanted)
        labels = convert(raw, mappings[name]['reference_to_mp4_xy'])
        geometry[name] = dict(unscaled_past_centers_outside=outside({t: f for t, f in raw.items()
            if any(st <= t < st+study.OBS for st in s['starts'])}, s['raster']),
            scaled_past_centers_outside=outside({t: f for t, f in labels.items()
            if any(st <= t < st+study.OBS for st in s['starts'])}, s['raster']))
        if probe['windows'][scene]['video'] == video:
            scale_labels[scene] = labels
        sources[name] = {}
        for a, lookup in lookups.items():
            c, packed = Counter(), []
            for st in s['starts']:
                w = read_window(lookup, a, scene, video, st)
                count, _, pairs = study.paired_window(w, labels)
                diag, _ = coverage.diagnostic(w, labels)
                c.update(count)
                c.update({k: v for k, v in diag.items() if k.startswith('publisher_ids_')})
                packed.extend(pairs)
            c['recordings_with_support'] = int(c['complete'] >= 8 and c['windows_with_complete'] >= 3)
            p = study.PRIVATE / 'raster_registered' / a / scene / f'{video}.npz'
            study.save_pairs(p, packed)
            totals[a].update(c)
            by_scene[scene][a].update(c)
            sources[name][a] = dict(c, paired_cache_sha256=study.sha(p))
            if packed:
                clean = np.array([[labels[st+j][i][0] for j in range(study.OBS)] for st, _, i, *_ in packed])
                camera = np.stack([p[3] for p in packed])
                center = lambda x: (x[:, :, :2]+x[:, :, 2:4])/2.
                f = np.stack((motion.forecasts(center(camera)), motion.forecasts(center(clean))), axis=1)
                y, v, m = (np.stack([p[k] for p in packed]) for k in (5, 6, 7))
                errors[a].append((np.linalg.norm(f-y[:, None, None], axis=-1), v, m, [(name, p[2]) for p in packed]))
    by_motion = {}
    for a, rows in errors.items():
        e, v, m, ids = zip(*rows, strict=True)
        by_motion[a] = motion.report(np.concatenate(e), np.concatenate(v), np.concatenate(m), sum(ids, []))
    scale_counts = {}
    for scene, r in probe['windows'].items():
        scale_counts[scene] = {}
        for arm, path in (('whole', study.PRIVATE/'rapid'/scene/r['video']/f"{r['start']:06}.json"),
                          ('tiled', scale_probe.PRIVATE/f'{scene}.json')):
            if study.sha(path) != r[arm]:
                raise ValueError('Scale probe cache changed')
            c, _, _ = study.paired_window(study.load(path), scale_labels[scene])
            scale_counts[scene][arm] = c
    study.immutable(OUT/'readout.json', dict(source='fresh_run', registration=study.sha(receipt),
        earlier_unscaled_match_invalid=True, no_detection_changes=True, scenes=by_scene, totals=totals,
        sources=sources, out_of_raster=geometry, motion=by_motion, scale_probe=scale_counts,
        scale_probe_freeze_sha256=study.sha(scale_probe.FREEZE), independent_confirmation=False,
        trained_model=False, policy_changed=False))
    print(dict(totals=totals, geometry={k:sum(v[k] for v in geometry.values()) for k in next(iter(geometry.values()))},
               ade={a: x['complete_ade_px'] for a, x in by_motion.items()}), flush=True)


if __name__ == '__main__':
    run()
