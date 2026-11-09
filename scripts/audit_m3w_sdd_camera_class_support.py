"""Measure class-resolved outcome support without replacing pedestrian matches."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_sdd_camera_full_roster as full
from scripts import audit_m3w_sdd_camera_motion as motion
from scripts import audit_m3w_sdd_camera_raster as raster
from scripts import audit_m3w_sdd_camera_unknown as unknown
from scripts import run_m3w_sdd_camera_confidence as confidence
from scripts import run_m3w_sdd_camera_support as study

np = study.np
OUT = study.PUBLIC/'class_support_v2'
PEDESTRIAN = 'Pedestrian'


def matches(window, labels, classes):
    people, other = [], []
    for frame in window['frames']:
        rows = np.asarray(frame['rows'], dtype=float).reshape(-1, 7)
        y = labels.get(frame['frame'], {})
        ped = study.associate(rows, {k: v for k, v in y.items() if classes[k] == PEDESTRIAN})
        busy = {i for i, _ in ped}
        free = rows[~np.isin(rows[:, 0], list(busy))]
        people.append(ped)
        other.append(study.associate(free, {k: v for k, v in y.items() if classes[k] != PEDESTRIAN}))
    paired = set.intersection(*people)
    added = set.intersection(*other)
    roster = {row[0] for row in window['forecasts']}
    if not {i for i, _ in paired | added} <= roster or len(dict(paired | added)) != len(paired | added):
        raise ValueError('Association must preserve a unique subset of the online roster')
    return dict(paired), dict(added)


def outcomes(labels, frames, agent):
    z = [labels.get(t, {}).get(agent) for t in frames]
    return (np.asarray([.5*(v[0][:2]+v[0][2:]) if v else (0., 0.) for v in z]),
            np.asarray([v is not None for v in z]), np.asarray([bool(v and v[1]) for v in z]))


def score_window(window, labels, classes, recording):
    pedestrian, additional = matches(window, labels, classes)
    expected = {row[1]: row[2] for row in study.paired_window(window, {
        t: {k: v for k, v in y.items() if classes[k] == PEDESTRIAN}
        for t, y in labels.items() if window['start'] <= t < window['start']+study.OBS+study.FUTURE
    })[2]}
    if pedestrian != expected:
        raise ValueError('Original pedestrian identities changed')
    histories = confidence.histories(window)
    h = np.asarray(list(histories.values())).reshape(-1, study.OBS, 6)
    predictions = motion.forecasts((h[:, :, :2]+h[:, :, 2:4])/2)
    ref = np.asarray([row[1:] for row in window['forecasts']]).reshape(-1, 2, study.FUTURE, 2)
    np.testing.assert_allclose(predictions[:, :2], ref, rtol=0, atol=1e-10)
    shape = len(h), study.FUTURE
    ys = np.zeros((*shape, 2))
    valid = np.zeros(shape, bool)
    by_class = defaultdict(list)
    start = window['start']
    pairs = pedestrian | additional
    for j, k in enumerate(histories):
        if k not in pairs:
            continue
        agent = pairs[k]
        future, mask, manual = outcomes(labels, range(start+study.OBS, start+study.OBS+study.FUTURE), agent)
        ys[j], valid[j] = future, mask
        past, available, _ = outcomes(labels, range(start, start+study.OBS), agent)
        if not available.all():
            raise ValueError('Stable match is missing its publisher history')
        pub = motion.forecasts(past[None])[0]
        error = np.linalg.norm(np.stack((predictions[j], pub))-future, axis=-1)
        by_class[classes[agent]].append((error, mask, manual, (recording, agent)))
    return (predictions, ys, valid, np.full(len(h), 1/max(len(h), 1))), by_class


def counts(rows):
    windows = [set(k for k, r in classes.items() if any(v[1].all() for v in r)) for classes in rows]
    by_class = {}
    for k in sorted({k for w in rows for k in w}):
        values = [v for w in rows for v in w.get(k, ())]
        err, mask, manual, keys = zip(*values, strict=True)
        by_class[k] = motion.report(np.asarray(err), np.asarray(mask), np.asarray(manual), keys)
        by_class[k]['complete_windows'] = sum(k in w for w in windows)
        by_class[k]['visible_future_steps'] = int(np.sum(mask))
        by_class[k]['non_generated_future_steps'] = int(np.sum(manual))
    return by_class


def window_support(values, group):
    n = len(values[2])
    complete = {k: sum(bool(v[1].all()) for v in rows) for k, rows in group.items()}
    total = sum(complete.values())
    return Counter(online_at_least_two=int(n >= 2),
                   complete_at_least_one=int(total >= 1),
                   complete_at_least_two=int(total >= 2),
                   complete_all_online_at_least_two=int(n >= 2 and total == n),
                   complete_at_least_two_classes=int(sum(v > 0 for v in complete.values()) >= 2))


def run():
    d, mapping, original = confidence.sources()
    files = [confidence.OUT/p for p in ('registration.json', 'freeze.json', 'readout.json', 'full_roster_utility.json')]
    reg, frozen, coverage, previous = map(study.load, files)
    entries = {(w['scene'], w['video'], w['start']): w['sha256'] for w in frozen['windows']}
    if (reg['code_sha256'] != study.sha(Path(confidence.__file__))
            or reg['raster'] != study.sha(raster.OUT/'registration.json')
            or frozen['registration'] != study.sha(files[0]) or frozen['labels_read']
            or not frozen['original_high_confidence_exact'] or not frozen['first_window_exact']
            or coverage['freeze_sha256'] != study.sha(files[1]) or entries.keys() != original.keys()
            or any(previous['parents'][p.name] != study.sha(p) for p in files[:3])):
        raise ValueError('Expected the exact label-blind confidence bank and original readouts')
    totals, vectors, scenes, clips = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list)), {}
    joint = defaultdict(Counter)
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        name = f'{scene}/{video}'
        _, ann = study.files(scene, video)
        if study.sha(ann) != s['annotation_sha256']:
            raise ValueError('Frozen annotation file changed')
        wanted = {t for st in s['starts'] for t in range(st, st+study.OBS+study.FUTURE)}
        labels, classes = unknown.annotations(ann, wanted, mapping['mappings'][name]['reference_to_mp4_xy'])
        by_arm = defaultdict(list)
        clip_joint = defaultdict(Counter)
        for start in s['starts']:
            p = confidence.HOME/scene/video/f'{start:06}.json'
            if study.sha(p) != entries[(scene, video, start)]:
                raise ValueError('Frozen observation bank changed')
            frames = study.load(p)['frames']
            hi, lo = (confidence.tracks(frames, q) for q in (confidence.HIGH, confidence.LOW))
            for arm, window in dict(high=hi, low=lo, bridge=confidence.bridge(hi, lo)).items():
                values, group = score_window(window, labels, classes, name)
                by_arm[arm].append(group)
                vectors[arm].append(values)
                clip_joint[arm].update(window_support(values, group))
        clips[name] = {}
        for arm, windows in by_arm.items():
            counts_here = counts(windows)
            if counts_here.get(PEDESTRIAN, {}).get('complete', 0) != coverage['records'][name][arm].get('complete', 0):
                raise ValueError('Published pedestrian support changed')
            totals[arm].extend(windows)
            scenes[scene][arm].extend(windows)
            joint[arm].update(clip_joint[arm])
            clips[name][arm] = dict(classes={k: {m: v[m] for m in (
                'rows', 'complete', 'complete_windows', 'distinct_recording_tracks', 'non_generated_future_steps'
            )} for k, v in counts_here.items()}, window_support=clip_joint[arm])
    combined = {}
    for arm, windows in totals.items():
        utility = full.readout(*(np.concatenate(x) for x in zip(*vectors[arm], strict=True)))
        categories = counts(windows)
        count = sum(x['complete'] for x in categories.values())
        if count != utility['complete'] or utility['online'] != coverage['totals'][arm]['online_histories']:
            raise ValueError('Class totals or online roster changed')
        supporting = [k for k, a in clips.items() if a[arm]['window_support']['complete_at_least_one'] >= 3
                      and sum(v['complete'] for v in a[arm]['classes'].values()) >= 8]
        by_class = Counter()
        for a in clips.values():
            for name, v in a[arm]['classes'].items():
                by_class[name] += int(v['complete'] >= 8 and v['complete_windows'] >= 3)
        combined[arm] = dict(classes=categories, full_roster_utility=utility, supporting_clips=supporting,
                             supporting_scenes=sorted({k.split('/')[0] for k in supporting}),
                             class_specific_supporting_clips=by_class, window_support=joint[arm],
                             clips_with_a_fully_labeled_joint_window=[
                                 k for k, a in clips.items() if a[arm]['window_support']['complete_all_online_at_least_two']])
    OUT.mkdir(parents=True, exist_ok=True)
    study.immutable(OUT/'readout.json', dict(source='fresh_run', inputs='cached_verified',
        code_sha256=study.sha(Path(__file__)), parents={p.name: study.sha(p) for p in files},
        association='reserve per-frame pedestrian matches, then associate remaining rows to other classes',
        class_used_as_input=False, role='same SDD development windows; no new predictions or fitting',
        unit='video pixel, raw frames', original_pedestrian_matches_preserved=True,
        independent_confirmation=False, policy_changed=False, arms=combined, clips=clips,
        scenes={s: {a: counts(rows) for a, rows in x.items()} for s, x in scenes.items()}))
    for arm, m in combined.items():
        print(arm, {k: v['complete'] for k, v in m['classes'].items()}, m['supporting_clips'],
              m['full_roster_utility']['contrasts']['ols8_vs_ols4'], flush=True)


if __name__ == '__main__':
    run()
