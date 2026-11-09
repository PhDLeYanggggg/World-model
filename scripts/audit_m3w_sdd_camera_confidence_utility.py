"""Recompute missing-outcome-aware motion contrasts for each confidence arm."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_sdd_camera_full_roster as full
from scripts import audit_m3w_sdd_camera_motion as motion
from scripts import audit_m3w_sdd_camera_raster as raster
from scripts import run_m3w_sdd_camera_confidence as confidence
from scripts import run_m3w_sdd_camera_support as study

np = study.np


def roster(window, labels):
    history = confidence.histories(window)
    _, _, packed = study.paired_window(window, labels)
    matched = {row[1]: row for row in packed}
    if not matched.keys() <= history.keys():
        raise ValueError('Matched rows must belong to the online roster')
    h = np.asarray(list(history.values())).reshape(-1, study.OBS, 6)
    f = motion.forecasts((h[:, :, :2]+h[:, :, 2:4])/2.)
    ref = np.asarray([row[1:] for row in window['forecasts']]).reshape(-1, 2, study.FUTURE, 2)
    np.testing.assert_allclose(f[:, :2], ref, rtol=0, atol=1e-10)
    y = np.zeros((len(h), study.FUTURE, 2))
    valid = np.zeros((len(h), study.FUTURE), dtype=bool)
    for j, k in enumerate(history):
        if k in matched:
            y[j], valid[j] = matched[k][5:7]
    return (f, y, valid, np.full(len(h), 1./max(len(h), 1))), matched


def run():
    d, mapping, original = confidence.sources()
    files = [confidence.OUT/p for p in ('registration.json', 'freeze.json', 'readout.json')]
    reg, frozen, scored = map(study.load, files)
    entries = {(w['scene'], w['video'], w['start']): w['sha256'] for w in frozen['windows']}
    if (reg['code_sha256'] != study.sha(Path(confidence.__file__))
            or reg['raster'] != study.sha(raster.OUT/'registration.json')
            or frozen['registration'] != study.sha(files[0]) or frozen['labels_read']
            or not frozen['original_high_confidence_exact'] or not frozen['first_window_exact']
            or scored['freeze_sha256'] != study.sha(files[1]) or entries.keys() != original.keys()):
        raise ValueError('Expected the verified confidence bank and its support readout')
    groups = defaultdict(list)
    effects = Counter()
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        _, ann = study.files(scene, video)
        if study.sha(ann) != s['annotation_sha256']:
            raise ValueError('Frozen annotation file changed')
        wanted = {t for st in s['starts'] for t in range(st, st+study.OBS+study.FUTURE)}
        labels = raster.convert(study.label_geometry(ann, wanted),
                                mapping['mappings'][f'{scene}/{video}']['reference_to_mp4_xy'])
        for start in s['starts']:
            p = confidence.HOME/scene/video/f'{start:06}.json'
            if study.sha(p) != entries[(scene, video, start)]:
                raise ValueError('Frozen detector observations changed')
            frames = study.load(p)['frames']
            hi, lo = (confidence.tracks(frames, q) for q in (confidence.HIGH, confidence.LOW))
            b = confidence.bridge(hi, lo)
            matched = {}
            for arm, w in dict(high=hi, low=lo, bridge=b).items():
                values, matched[arm] = roster(w, labels)
                groups[arm].append(values)
            retained = {row[0] for row in hi['forecasts']}
            for i in retained:
                old, new = (matched[a].get(i) for a in ('high', 'bridge'))
                effects['retained_history_outcome_identity_changed'] += int(
                    (old[2] if old else None) != (new[2] if new else None))
            for i, *_ in b['forecasts']:
                if i not in retained:
                    row = matched['bridge'].get(i)
                    effects['added_histories'] += 1
                    effects['added_complete'] += int(row is not None and row[6].all())
                    effects['added_unknown'] += int(row is None or not row[6].any())
    result = {a: full.readout(*(np.concatenate(v) for v in zip(*rows, strict=True)))
              for a, rows in groups.items()}
    for a, m in result.items():
        reference = scored['totals'][a]
        if [m[k] for k in ('online', 'complete', 'wholly_unknown')] != [
                reference[k] for k in ('online_histories', 'complete', 'wholly_unknown')]:
            raise ValueError('Utility calculation must preserve the support readout roster')
    high = study.load(raster.OUT/'full_online_readout.json')['arms']['aerial']['all']
    for key in high['contrasts']:
        for stat in high['contrasts'][key]:
            np.testing.assert_allclose(result['high']['contrasts'][key][stat],
                                       high['contrasts'][key][stat], rtol=0, atol=1e-10)
    study.immutable(confidence.OUT/'full_roster_utility.json', dict(
        source='fresh_run', inputs='cached_verified', code_sha256=study.sha(Path(__file__)),
        parents={p.name: study.sha(p) for p in files}, metrics=result, bridge_effects=effects,
        role='development diagnostic; within-arm motion contrasts on different online rosters',
        weighting='equal over nonempty online windows, normalized by observed CV error mass',
        independent_confirmation=False, policy_changed=False))
    print({a: m['contrasts']['ols8_vs_ols4'] for a, m in result.items()}, dict(effects), flush=True)


if __name__ == '__main__':
    run()
