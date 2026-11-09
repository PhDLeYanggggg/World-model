"""Upper-bound observable histories without imposing unique box assignments."""
from collections import Counter, defaultdict
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_aerialmpt_camera as camera

study, np = camera.study, camera.np


def possible_histories(frames, labels, radius, confidence):
    possible = []
    for t, members in enumerate(frames):
        ids = sorted(labels.get(t, {}))
        boxes = np.asarray(members, dtype=float).reshape(-1, 6)
        boxes = boxes[boxes[:, 4] >= confidence]
        if not ids or not len(boxes):
            possible.append(set())
            continue
        centers = .5*(boxes[:, :2]+boxes[:, 2:4])
        ref = np.stack([labels[t][k] for k in ids])
        d = np.linalg.norm(ref[:, None] - centers[None], axis=-1)
        possible.append({ids[i] for i in np.flatnonzero((d <= radius).any(1))})
    if len(possible) != camera.OBS:
        raise ValueError('Require all eight observation frames')
    history = set.intersection(*possible)
    future = set.intersection(*(set(labels.get(t, {})) for t in range(camera.OBS, camera.OBS+camera.FUTURE)))
    return dict(histories=len(history), histories_with_complete_future=len(history & future))


def run():
    with ZipFile(camera.SOURCE) as archive:
        reg, key = camera.registration(archive)
        fpath = camera.OUT/'freeze.json'
        f = study.load(fpath)
        if f['registration'] != key or f['label_rows_read'] or f['recordings'].keys() != reg['windows'].keys():
            raise ValueError('Observation freeze mismatch')
        result = {}
        totals = defaultdict(Counter)
        for clip, source in reg['windows'].items():
            path = camera.PRIVATE/f'{clip}.json'
            if study.sha(path) != f['recordings'][clip]:
                raise ValueError('Frozen detection bank changed')
            d = study.load(path)['frames']
            xy = {k: [frame['boxes'] for frame in v] for k, v in d.items()}
            xy['union'] = [a+b for a, b in zip(xy['grid2'], xy['grid4'], strict=True)]
            targets = camera.labels(archive, clip, source['frames'])
            result[clip] = {}
            for arm, frames in xy.items():
                for score in (.30, .12):
                    for radius in camera.RADII:
                        name = f'{arm}_score{score:.2f}_radius{radius:.0f}'
                        count = possible_histories(frames, targets, radius, score)
                        result[clip][name] = count
                        totals[name].update(count)
    study.immutable(camera.OUT/'detection_ceiling.json', dict(source='fresh_run',
        freeze=study.sha(fpath), code=study.sha(Path(__file__)), role='development diagnostic',
        bound='each reference needs a nearby frozen detection on every observed step; reuse of detections allowed',
        no_claim_for_gap_filling=True, clips=result, totals=totals))
    print({k: v for k, v in totals.items() if k.endswith('radius4')})


if __name__ == '__main__':
    run()
