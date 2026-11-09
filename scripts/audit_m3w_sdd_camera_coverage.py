"""Separate missed detections, broken track identities and unavailable futures."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_sdd_camera_aerial as aerial
from scripts import run_m3w_sdd_camera_support as study

np = study.np
BINS = ('under_10', '10_to_under_20', '20_to_under_40', '40_or_more')


def diagnostic(w, labels):
    count = Counter(windows=1)
    bins = defaultdict(Counter)
    sets = []
    for f in w['frames']:
        a = np.asarray(f['rows'], dtype=float).reshape(-1, 7)
        ref = labels.get(f['frame'], {})
        matched = study.associate(a, ref)
        people = {j for _, j in matched}
        sets.append(matched)
        count['frames'] += 1
        count['frames_with_no_detections'] += int(not len(a))
        count['frames_with_no_associations'] += int(not matched)
        for p, (box, _) in ref.items():
            k = BINS[int(np.searchsorted((10, 20, 40), min(box[2:] - box[:2]), side='right'))]
            bins[k]['publisher_boxes'] += 1
            bins[k]['matched_boxes'] += int(p in people)
    all_frames = set.intersection(*({j for _, j in s} for s in sets))
    continuous = set.intersection(*sets)
    count['publisher_ids_matched_at_every_frame'] = len(all_frames)
    count['publisher_ids_lost_to_id_changes'] = len(all_frames) - len(continuous)
    count['stable'] = len(continuous)
    return count, bins


def run():
    d, reg = study.checked_registration()
    main = study.load(study.PUBLIC / 'readout.json')
    extra = study.load(aerial.PUBLIC / 'readout.json')
    if main['registration'] != reg or study.load(aerial.REG)['parent'] != reg:
        raise ValueError('Coverage audit requires the shared frozen camera roster')
    refs = {a: study.PUBLIC / f'{a}_freeze.json' for a in study.ARMS} | {aerial.ARM: aerial.FREEZE}
    h = dict(main['freezes'], aerial=extra['freeze'])
    if any(study.sha(p) != h[a] for a, p in refs.items()):
        raise ValueError('A bound camera freeze changed')
    entries = {a: {(w['scene'], w['video'], w['start']): w['sha256']
                   for w in study.load(p)['windows']} for a, p in refs.items()}
    totals = {a: Counter() for a in refs}
    size = {a: defaultdict(Counter) for a in refs}
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        _, ann = study.files(scene, video)
        if study.sha(ann) != s['annotation_sha256']:
            raise ValueError('Bound annotation payload changed')
        wanted = {t for start in s['starts'] for t in range(start, start+study.OBS)}
        labels = study.label_geometry(ann, wanted)
        for start in s['starts']:
            for a, lookup in entries.items():
                path = study.PRIVATE / a / scene / video / f'{start:06}.json'
                if study.sha(path) != lookup[(scene, video, start)]:
                    raise ValueError('Unbound detection file')
                c, b = diagnostic(study.load(path), labels)
                totals[a].update(c)
                for k, v in b.items():
                    size[a][k].update(v)
    for a in refs:
        r = extra['total'] if a == aerial.ARM else main['totals'][a]
        if (totals[a]['stable'] != r['stable'] or sum(b['publisher_boxes'] for b in size[a].values()) != r['label_boxes']
                or sum(b['matched_boxes'] for b in size[a].values()) != r['frame_matches']):
            raise ValueError('Coverage counts disagree with support readout')
        totals[a]['stable_with_incomplete_future'] = r['stable'] - r.get('complete', 0)
    study.immutable(study.PUBLIC / 'coverage_diagnostics.json', dict(registration=reg, freezes=h,
        source='fresh_run', box_size='shorter side of publisher box in pixels',
        implementation_sha256=study.sha(Path(__file__)), totals=totals, size=size))
    print(dict(totals=totals, size=size), flush=True)


if __name__ == '__main__':
    run()
