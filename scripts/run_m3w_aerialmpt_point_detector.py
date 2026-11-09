"""Fit a small location-held-out point detector on AerialMPT training clips."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_aerialmpt_detection_ceiling as ceiling
from scripts import run_m3w_aerialmpt_camera as camera

study, np, torch, cv2 = camera.study, camera.np, camera.torch, camera.cv2
nn, F = torch.nn, torch.nn.functional
OUT = camera.OUT.parent/'aerialmpt_point_detector_v1'
PRIVATE = camera.PRIVATE.parent/OUT.name
SEEDS = (17, 29, 43)
STEPS, BATCH, PATCH = 120, 6, 80
CUTS = (.10, .20, .35, .50, .65, .80, .90)
STRIDE = 2


def group(clip):
    return next((x for x in ('bauma', 'pasing') if clip.startswith(x)), clip)


def plans(clips, held):
    locations = sorted({group(k) for k in clips})
    folds = {}
    for target in sorted({group(k) for k in held}):
        calibration = locations[(locations.index(target)+1) % len(locations)]
        folds[target] = dict(
            fit=sorted(k for k in clips if group(k) not in (target, calibration)),
            calibrate=sorted(k for k in clips if group(k) == calibration),
            evaluate=sorted(k for k in held if group(k) == target))
    return folds


def register(archive):
    frozen, parent = camera.registration(archive)
    clip_images = {}
    for k in sorted(archive.namelist()):
        m = re.fullmatch(r'train/([A-Za-z0-9_]+)/image_list\.txt', k)
        if m:
            clip_images[m[1]] = [f'train/{m[1]}/{i}.png' for i in archive.read(k).decode().split()]
    reg = OUT/'registration.json'
    study.immutable(reg, dict(code=study.sha(Path(__file__)), parent=parent,
        old_readout=study.sha(camera.OUT/'readout.json'), folds=plans(clip_images, frozen['windows']),
        seeds=SEEDS, steps=STEPS, batch=BATCH, patch=PATCH, stride=STRIDE, candidate_thresholds=CUTS,
        validation_rule='maximize 4px point F1 on separate donor location, tie to higher threshold',
        input='one RGB image, no neighboring frames or labels', role='publisher-train development cross-fitting',
        no_publisher_test=True, workers=0, torch_threads=4))
    return study.load(reg), study.sha(reg), clip_images, frozen['windows']


def decode(archive, member):
    im = cv2.imdecode(np.frombuffer(archive.read(member), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError('Missing image')
    return np.ascontiguousarray(im[..., ::-1])


def read_points(archive, clip, length, limit):
    table = defaultdict(dict)
    with archive.open(f'train/{clip}/{clip}_gts.txt') as fh:
        for row in fh:
            t, agent, x, y, w, h, one, a, b = row.decode().split()
            t, agent = int(t)-1, int(agent)
            if not 0 <= t < length or (w, h, one, a, b) != ('4','4','1','-1','-1'):
                raise ValueError('Invalid release annotation')
            if t < limit:
                xy = np.array([float(x)+2, float(y)+2])
                if agent in table[t] or not np.isfinite(xy).all():
                    raise ValueError('Duplicate or nonfinite annotation')
                table[t][agent] = xy
    return table


def image_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float().div_(127.5).sub_(1.)


def model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=STRIDE, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
        nn.Conv2d(16, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
        nn.Conv2d(24, 24, 3, dilation=2, padding=2), nn.GroupNorm(4, 24), nn.SiLU(),
        nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
        nn.Conv2d(24, 1, 1))


def patches(donors, rng):
    pixels, maps = [], []
    for _ in range(BATCH):
        im, points = donors[rng.integers(len(donors))]
        h, w = im.shape[:2]
        x, y = rng.integers(w-PATCH+1), rng.integers(h-PATCH+1)
        image = im[y:y+PATCH, x:x+PATCH]
        p = points-[x, y]
        p = p[((p >= 0.) & (p < PATCH)).all(1)]
        if rng.random() < .5:
            image = image[:, ::-1]
            p[:, 0] = PATCH-1-p[:, 0]
        if rng.random() < .5:
            image = image[::-1]
            p[:, 1] = PATCH-1-p[:, 1]
        ij = np.rint(p/STRIDE).astype(int).clip(0, PATCH//STRIDE-1)
        target = np.zeros((PATCH//STRIDE, PATCH//STRIDE), np.float32)
        target[ij[:, 1], ij[:, 0]] = 1
        pixels.append(image)
        maps.append(target)
    return image_tensor(np.stack(pixels)), torch.from_numpy(np.stack(maps))[:, None]


def donor_images(archive, clip, members, cap=None):
    gt = read_points(archive, clip, len(members), len(members))
    indices = range(len(members)) if cap is None else np.rint(np.linspace(0, len(members)-1, cap)).astype(int)
    return [(decode(archive, members[i]), gt[i]) for i in indices]


def checkpoint(path, value):
    tmp = path.with_suffix('.pending')
    torch.save(value, tmp)
    tmp.replace(path)


def fit(donors, seed, path, contract, fold):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = model()
    opt = torch.optim.AdamW(net.parameters(), lr=.002)
    start, logs = 0, []
    if path.exists():
        s = torch.load(path, map_location='cpu', weights_only=True)
        if s['registration'] != contract:
            raise ValueError('Checkpoint protocol changed')
        net.load_state_dict(s['model'])
        opt.load_state_dict(s['optimizer'])
        rng.bit_generator.state = s['rng']
        start, logs = s['step'], s['loss']
    for i in range(start, STEPS):
        x, y = patches(donors, rng)
        logits = net(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(40.))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i+1) % 40 == 0:
            logs.append(float(loss.detach()))
            checkpoint(path, dict(registration=contract, model=net.state_dict(), optimizer=opt.state_dict(),
                                  rng=rng.bit_generator.state, step=i+1, loss=logs))
            status = dict(pid=os.getpid(), fold=fold, seed=seed, done=i+1, total=STEPS, loss=logs[-1])
            heartbeat = PRIVATE/'heartbeat.pending'
            heartbeat.write_text(json.dumps(status)+'\n')
            heartbeat.replace(PRIVATE/'heartbeat.json')
            print(status, flush=True)
    return net.eval(), logs


@torch.inference_mode()
def points(net, im):
    prob = net(image_tensor(im[None])).sigmoid()
    peaks = (prob == F.max_pool2d(prob, 3, stride=1, padding=1)) & (prob >= CUTS[0])
    y, x = torch.where(peaks[0, 0])
    xx, yy = (STRIDE*x).numpy(), (STRIDE*y).numpy()
    p = prob[0, 0, y, x].numpy()
    # Synthetic square neighborhoods control the unchanged association rule.
    return np.column_stack((xx-4, yy-4, xx+4, yy+4, p, np.ones(len(x))))


def choose_cut(net, validation):
    report = {c: dict(match=0, predictions=0, labels=0) for c in CUTS}
    for im, target in validation:
        candidates = points(net, im)
        for c, value in report.items():
            z = candidates[candidates[:, 4] >= c]
            rows = np.column_stack((np.arange(len(z)), z))
            value['match'] += len(camera.associate(rows, target, 4.))
            value['predictions'] += len(z)
            value['labels'] += len(target)
    for d in report.values():
        d['point_f1'] = 2*d['match']/max(d['predictions']+d['labels'], 1)
    return max(CUTS, key=lambda c: (report[c]['point_f1'], c)), {str(k): v for k, v in report.items()}


def freeze(archive, reg, key, images):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    f = {}
    for location, fold in reg['folds'].items():
        donors = [(im, np.asarray(list(pts.values())).reshape(-1, 2)) for k in fold['fit']
                  for im, pts in donor_images(archive, k, images[k])]
        validation = [z for k in fold['calibrate'] for z in donor_images(archive, k, images[k], cap=6)]
        f[location] = {}
        for seed in SEEDS:
            weight = PRIVATE/f'{location}_{seed}.pt'
            net, losses = fit(donors, seed, weight, key, location)
            cut, val = choose_cut(net, validation)
            named = {}
            for clip in fold['evaluate']:
                frames = []
                for t, name in enumerate(images[clip][:camera.OBS]):
                    im = decode(archive, name)
                    a = points(net, im)
                    if t == 0:
                        np.testing.assert_array_equal(a, points(net, im))
                    frames.append(dict(frame=t, pixel_sha256=study.mobile.file_digest_array(im), boxes=a.tolist()))
                path = PRIVATE/f'{clip}_{seed}.json'
                study.immutable(path, dict(registration=key, cut=cut, frames=frames, held_labels_read=False))
                named[clip] = study.sha(path)
            f[location][str(seed)] = dict(cut=cut, validation=val, loss=losses, weights=study.sha(weight), windows=named)
    study.immutable(OUT/'freeze.json', dict(registration=key, fits=f, in_fold_held_labels_read=False))


def score(archive, reg, key, held):
    f = study.load(OUT/'freeze.json')
    if f['registration'] != key or f['in_fold_held_labels_read'] or f['fits'].keys() != reg['folds'].keys():
        raise ValueError('Scoring requires the full frozen cross-fitting experiment')
    result = defaultdict(dict)
    for location, by_seed in f['fits'].items():
        for seed, a in by_seed.items():
            for clip, sha in a['windows'].items():
                path = PRIVATE/f'{clip}_{seed}.json'
                if study.sha(path) != sha or group(clip) != location:
                    raise ValueError('Wrong held observation bank')
                z = study.load(path)
                if z['registration'] != key or z['cut'] != a['cut'] or z['held_labels_read']:
                    raise ValueError('Held input provenance changed')
                label = camera.labels(archive, clip, held[clip]['frames'])
                window = camera.confidence.tracks(z['frames'], z['cut'])
                record = camera.assess(window, label, 4.)
                record['detection_ceiling'] = ceiling.possible_histories(
                    [u['boxes'] for u in z['frames']], label, 4., z['cut'])['histories_with_complete_future']
                result[seed][clip] = record
    totals = {}
    for seed, cells in result.items():
        if cells.keys() != held.keys():
            raise ValueError('Incomplete held clip set')
        totals[seed] = {k: sum(x[k] for x in cells.values()) for k in (
            'reference_complete', 'matched_points', 'label_points', 'online', 'stable', 'complete',
            'fully_labeled_joint', 'detection_ceiling')}
    study.immutable(OUT/'readout.json', dict(source='fresh_run', freeze=study.sha(OUT/'freeze.json'),
        totals=totals, results=result, test_members_read=False, independent_confirmation=False, policy_changed=False))
    print(totals, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    args = p.parse_args()
    for path in (OUT, PRIVATE):
        path.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock, ZipFile(camera.SOURCE) as archive:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg, key, images, held = register(archive)
        if args.phase == 'freeze':
            freeze(archive, reg, key, images)
        else:
            score(archive, reg, key, held)
