"""Recalibrate fixed detector checkpoints for stable eight-step histories."""
from collections import Counter, defaultdict
import fcntl
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_aerialmpt_point_detector as point

study, torch = point.study, point.torch
OUT = point.OUT/'track_guard'
MIN_PRECISION, MIN_MATCHES = .5, 2


def choose(calibration, floor):
    legal = [c for c, stat in calibration.items()
             if c >= floor and stat['stable'] >= MIN_MATCHES and stat['stable'] >= MIN_PRECISION*stat['online']]
    return max(legal, key=lambda c: (calibration[c]['stable'], c), default=None)


def run():
    with (point.PRIVATE/'process.lock').open('a') as lock, ZipFile(point.camera.SOURCE) as archive:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg, parent, images, held = point.register(archive)
        fpath = point.OUT/'freeze.json'
        f = study.load(fpath)
        if f['registration'] != parent or f['in_fold_held_labels_read']:
            raise ValueError('Wrong point detector freeze')
        OUT.mkdir(parents=True, exist_ok=True)
        setup = OUT/'registration.json'
        study.immutable(setup, dict(code=study.sha(Path(__file__)), parent=study.sha(fpath),
            select_on='original donor calibration clips; first eight images; identity stability without future labels',
            precision_floor=MIN_PRECISION, stable_floor=MIN_MATCHES, abstain_if_unsupported=True,
            role='post-diagnostic development comparison, not an independent test'))
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        point.cv2.setNumThreads(1)
        torch.use_deterministic_algorithms(True)
        selected, proof = defaultdict(dict), defaultdict(dict)
        for location, fold in reg['folds'].items():
            calibration = [
                ([(t, point.decode(archive, n)) for t, n in enumerate(images[k][:point.camera.OBS])],
                 point.read_points(archive, k, len(images[k]), point.camera.OBS))
                for k in fold['calibrate']
            ]
            for seed, a in f['fits'][location].items():
                p = point.PRIVATE/f'{location}_{seed}.pt'
                if study.sha(p) != a['weights']:
                    raise ValueError('Point detector weights changed')
                state = torch.load(p, map_location='cpu', weights_only=True)
                net = point.model()
                net.load_state_dict(state['model'])
                net.eval()
                counts = {c: Counter() for c in point.CUTS}
                for frames, y in calibration:
                    bank = [dict(frame=t, pixel_sha256=study.mobile.file_digest_array(im), boxes=point.points(net, im).tolist())
                            for t, im in frames]
                    for c, count in counts.items():
                        w = point.camera.confidence.tracks(bank, c)
                        s = point.camera.assess(w, y, 4.)
                        count.update({k: s[k] for k in ('stable', 'online', 'reference_histories')})
                selected[location][seed] = choose(counts, a['cut'])
                proof[location][seed] = {str(k): v for k, v in counts.items()}
        policy = OUT/'policy.json'
        study.immutable(policy, dict(registration=study.sha(setup), cuts=selected, calibration=proof,
                                    held_labels_read=False))
        results, totals = defaultdict(dict), defaultdict(Counter)
        for location, arms in f['fits'].items():
            for seed, a in arms.items():
                cut = selected[location][seed]
                for clip, sha in a['windows'].items():
                    p = point.PRIVATE/f'{clip}_{seed}.json'
                    if study.sha(p) != sha:
                        raise ValueError('Point detector bank changed')
                    frames = study.load(p)['frames']
                    w = point.camera.confidence.tracks(frames, cut if cut is not None else 2.)
                    labels = point.camera.labels(archive, clip, held[clip]['frames'])
                    s = point.camera.assess(w, labels, 4.)
                    results[seed][clip] = dict(s, abstained=cut is None)
                    totals[seed].update({k: int(s[k]) for k in (
                        'online', 'stable', 'complete', 'wholly_unknown', 'fully_labeled_joint')})
        study.immutable(OUT/'readout.json', dict(source='fresh_run', policy=study.sha(policy),
            totals=totals, results=results, independent_confirmation=False, deployment_changed=False))
        print(dict(cuts=selected, totals=totals), flush=True)


if __name__ == '__main__':
    run()
