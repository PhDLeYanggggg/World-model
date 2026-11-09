"""Export all causal tracker histories, not just the annotation-matched subset."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_sdd_camera_support as study

np = study.np


def rows(w, matched):
    for online, cv, ols4 in w['forecasts']:
        hist = [next(r[1:] for r in frame['rows'] if r[0] == online) for frame in w['frames']]
        label = matched.get((w['start'], online))
        yield (w['start'], online, hist, (cv, ols4)), label


def export(arm, s, entries, readout):
    scene, video = s['scene'], s['video']
    paired = study.PRIVATE / 'paired' / arm / scene / f'{video}.npz'
    if study.sha(paired) != readout['paired_cache_sha256']:
        raise ValueError('Paired cache does not match the geometry audit')
    with np.load(paired, allow_pickle=False) as z:
        labels = {(int(t), int(i)): (y, v, m) for t, i, y, v, m in zip(
            z['start'], z['online_id'], z['target_xy'], z['target_mask'], z['non_generated_mask'], strict=True)}
    prefixes, targets, masks, origin, matched = [], [], [], [], []
    for start in s['starts']:
        p = study.PRIVATE / arm / scene / video / f'{start:06}.json'
        if study.sha(p) != entries[(scene, video, start)]:
            raise ValueError('Frozen detection payload changed')
        for prefix, label in rows(study.load(p), labels):
            prefixes.append(prefix)
            matched.append(label is not None)
            y, v, m = label if label is not None else (np.zeros((12, 2)), np.zeros(12, bool), np.zeros(12, bool))
            targets.append(y)
            masks.append(v)
            origin.append(m)
    if len(prefixes) != readout['online_histories'] or sum(matched) != readout['stable']:
        raise ValueError('Dropping unmatched online tracks would change the roster')
    home = study.PRIVATE / 'all_online' / arm / scene
    home.mkdir(parents=True, exist_ok=True)
    x, y = home / f'{video}_inputs.npz', home / f'{video}_outcomes.npz'
    frames, ids, hist, forecast = zip(*prefixes, strict=True) if prefixes else ((), (), (), ())
    private = {x: dict(start=np.asarray(frames, dtype='int64'), online_id=np.asarray(ids, dtype='int64'),
                       past_xyxy_conf_class=np.asarray(hist, dtype='float64').reshape(-1, 8, 6),
                       cv_ols4=np.asarray(forecast, dtype='float64').reshape(-1, 2, 12, 2)),
               y: dict(target_xy=np.asarray(targets, dtype='float64').reshape(-1, 12, 2),
                       target_mask=np.asarray(masks, dtype=bool).reshape(-1, 12),
                       non_generated_mask=np.asarray(origin, dtype=bool).reshape(-1, 12),
                       stable_match=np.asarray(matched, dtype=bool))}
    for path, arrays in private.items():
        tmp = path.with_suffix('.pending.npz')
        np.savez(tmp, **arrays)
        tmp.replace(path)
    return dict(input_sha256=study.sha(x), outcome_sha256=study.sha(y),
                online=len(prefixes), stable=sum(matched), wholly_unknown=sum(not v.any() for v in masks))


def run():
    d, reg = study.checked_registration()
    readout = study.PUBLIC / 'readout.json'
    parent = study.load(readout)
    if parent['registration'] != reg:
        raise ValueError('Require the matching frozen geometry audit')
    arms = {}
    for arm in study.ARMS:
        p = study.PUBLIC / f'{arm}_freeze.json'
        if study.sha(p) != parent['freezes'][arm]:
            raise ValueError('Camera freeze changed since geometry audit')
        entries = {(w['scene'], w['video'], w['start']): w['sha256'] for w in study.load(p)['windows']}
        recs, total = {}, Counter()
        for s in d['sources']:
            name = f"{s['scene']}/{s['video']}"
            recs[name] = r = export(arm, s, entries, parent['recordings'][name][arm])
            total.update({k: r[k] for k in ('online', 'stable', 'wholly_unknown')})
        arms[arm] = dict(recordings=recs, total=total)
    study.immutable(study.PUBLIC / 'donor_export.json', dict(source='fresh_run', registration=reg,
        support_readout_sha256=study.sha(readout), implementation_sha256=study.sha(Path(__file__)), arms=arms,
        development_only=True, inputs_outcomes_separate=True, fitted_policy=False, new_split=False))
    print({k: d['total'] for k, d in arms.items()}, flush=True)


if __name__ == '__main__':
    run()
