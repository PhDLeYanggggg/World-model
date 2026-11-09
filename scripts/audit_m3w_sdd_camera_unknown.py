"""Explain unknown SDD camera tracks without relaxing the scoring roster."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_sdd_camera_raster as raster
from scripts import run_m3w_sdd_camera_aerial as aerial
from scripts import run_m3w_sdd_camera_support as study

np = study.np


def annotations(path, wanted, scale):
    rows = defaultdict(dict)
    classes = {}
    with path.open() as stream:
        for line in stream:
            i, x0, y0, x1, y1, t, lost, hidden, generated, name = line.split()
            t = int(t)
            if t not in wanted or lost != '0' or hidden != '0':
                continue
            i = int(i)
            box = np.asarray([x0, y0, x1, y1], dtype=float)
            if (box[2:] <= box[:2]).any() or i in rows[t]:
                raise ValueError('Degenerate or duplicate annotation')
            name = name.strip('"')
            if classes.setdefault(i, name) != name:
                raise ValueError('Annotation class changed across the window')
            rows[t][i] = box * np.tile(scale, 2), generated == '0'
    return rows, classes


def reasons(window, labels, classes):
    frames = [np.asarray(f['rows'], dtype=float).reshape(-1, 7) for f in window['frames']]
    past = [labels.get(f['frame'], {}) for f in window['frames']]
    ped = [study.associate(a, {i: b for i, b in y.items() if classes[i] == 'Pedestrian'})
           for a, y in zip(frames, past, strict=True)]
    any_class = [study.associate(a, y) for a, y in zip(frames, past, strict=True)]
    stable_ped = dict(set.intersection(*ped))
    stable_all = dict(set.intersection(*any_class))
    count = Counter()
    stable_nonped_classes = Counter()
    for i, *_ in window['forecasts']:
        if i in stable_ped:
            person = stable_ped[i]
            n = sum(person in labels.get(t, {}) for t in range(window['start']+study.OBS,
                                                               window['start']+study.OBS+study.FUTURE))
            count['scoring_complete' if n == study.FUTURE else 'scoring_partial' if n else 'no_future'] += 1
        elif i in stable_all and classes[stable_all[i]] != 'Pedestrian':
            count['stable_nonpedestrian'] += 1
            stable_nonped_classes[classes[stable_all[i]]] += 1
        elif any(any(k == i for k, _ in s) for s in ped):
            count['intermittent_pedestrian_match'] += 1
        elif any(any(k == i for k, _ in s) for s in any_class):
            count['intermittent_other_class_match'] += 1
        else:
            count['never_matched_visible_label'] += 1
    return count, stable_nonped_classes


def run():
    d, reg = study.checked_registration()
    files = raster.OUT/'registration.json', raster.OUT/'readout.json', aerial.FREEZE
    m, r, f = map(study.load, files)
    if (m['parent'] != reg or r['registration'] != study.sha(files[0])
            or m['freezes']['aerial'] != study.sha(files[2]) or f['label_geometry_used']):
        raise ValueError('Require raster-registered labels on the frozen aerial roster')
    entries = {(w['scene'], w['video'], w['start']): w['sha256'] for w in f['windows']}
    total, classes, clips = Counter(), Counter(), {}
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        name = f'{scene}/{video}'
        _, ann = study.files(scene, video)
        if study.sha(ann) != s['annotation_sha256']:
            raise ValueError('Annotation payload changed')
        wanted = {t for st in s['starts'] for t in range(st, st+study.OBS+study.FUTURE)}
        labels, k = annotations(ann, wanted, m['mappings'][name]['reference_to_mp4_xy'])
        c, v = Counter(), Counter()
        for start in s['starts']:
            w = raster.read_window(entries, 'aerial', scene, video, start)
            z, k0 = reasons(w, labels, k)
            c.update(z)
            v.update(k0)
        ref = r['sources'][name]['aerial']
        if sum(c.values()) != ref['online_histories'] or c['scoring_complete'] != ref.get('complete', 0):
            raise ValueError('Explanatory audit must preserve the corrected scoring roster')
        clips[name] = dict(reasons=c, stable_nonped_classes=v)
        total.update(c)
        classes.update(v)
    study.immutable(raster.OUT/'unknown_taxonomy.json', dict(source='fresh_run',
        parents={str(p.relative_to(study.PUBLIC)): study.sha(p) for p in files}, code=study.sha(Path(__file__)),
        reasons=total, stable_nonped_classes=classes, clips=clips, scoring_labels_changed=False,
        role='existing development data only; never-matched is not synonymous with false detection'))
    print(dict(reasons=total, stable_nonped_classes=classes), flush=True)


if __name__ == '__main__':
    run()
