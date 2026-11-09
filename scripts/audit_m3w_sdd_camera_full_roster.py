"""Reattach raster-registered outcomes to all frozen online histories."""
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_sdd_camera_motion as motion
from scripts import audit_m3w_sdd_camera_raster as raster
from scripts import run_m3w_sdd_camera_aerial as aerial
from scripts import run_m3w_sdd_camera_support as study
from src.evaluation.m3w_opendd_location import paired

np = study.np


def attach(inputs, corrected):
    keys = lambda z: list(zip(z['start'].tolist(), z['online_id'].tolist(), strict=True))
    roster, matched = keys(inputs), keys(corrected)
    position = {key: i for i, key in enumerate(roster)}
    if len(position) != len(roster) or len(set(matched)) != len(matched) or not set(matched) <= position.keys():
        raise ValueError('Stable matches must be a unique subset of the frozen online roster')
    at = [position[k] for k in matched]
    for k in ('past_xyxy_conf_class', 'cv_ols4'):
        np.testing.assert_array_equal(inputs[k][at], corrected[k])
    out = {k: np.zeros((len(roster), *corrected[k].shape[1:]), dtype=corrected[k].dtype)
           for k in ('target_xy', 'target_mask', 'non_generated_mask')}
    for k in out:
        out[k][at] = corrected[k]
    out['stable_match'] = np.zeros(len(roster), dtype=bool)
    out['stable_match'][at] = True
    return out


def readout(f, y, valid, weight):
    mass = float(weight @ np.where(valid, np.linalg.norm(f[:, 0]-y, axis=-1), 0.).sum(1))
    contrasts = {}
    for j, name in enumerate(motion.METHODS[1:], 1):
        for k, ref in enumerate(motion.METHODS[:2]):
            if j == k:
                continue
            known, low = paired(f[:, k], f[:, j], y, valid)
            g, l = (float(weight @ v.sum(1)) for v in (known, low))
            contrasts[f'{name}_vs_{ref}'] = dict(known_pp=100*g/mass if mass > 0 else None,
                lower_pp=100*l/mass if mass > 0 else None, missing_debit_px=g-l)
    return dict(online=len(f), complete=int(valid.all(1).sum()), wholly_unknown=int((~valid.any(1)).sum()),
                known_cv_mass_px=mass, contrasts=contrasts)


def run():
    d, reg = study.checked_registration()
    files = [raster.OUT/'registration.json', raster.OUT/'readout.json',
             study.PUBLIC/'donor_export.json', aerial.PUBLIC/'readout.json']
    m, r, donors, ext = map(study.load, files)
    if (m['parent'] != reg or r['registration'] != study.sha(files[0]) or donors['registration'] != reg
            or m['freezes']['aerial'] != ext['freeze'] or not donors['inputs_outcomes_separate']):
        raise ValueError('Donor export and raster correction must derive from the same frozen videos')
    old = {a: z['recordings'] for a, z in donors['arms'].items()}
    old['aerial'] = {k: v['donor_export'] for k, v in ext['recordings'].items()}
    records, groups = {}, defaultdict(lambda: defaultdict(list))
    for s in d['sources']:
        scene, video = s['scene'], s['video']
        name = f'{scene}/{video}'
        records[name] = {}
        for a, origins in old.items():
            prefix = study.PRIVATE/'all_online'/a/scene/f'{video}_inputs.npz'
            match = study.PRIVATE/'raster_registered'/a/scene/f'{video}.npz'
            record = r['sources'][name][a]
            ih = study.sha(prefix)
            if ih != origins[name]['input_sha256'] or study.sha(match) != record['paired_cache_sha256']:
                raise ValueError('Changed frozen input or corrected outcome')
            with np.load(prefix, allow_pickle=False) as x, np.load(match, allow_pickle=False) as z:
                y = attach(x, z)
                hist = x['past_xyxy_conf_class']
                f = motion.forecasts((hist[:, :, :2]+hist[:, :, 2:4])/2.)
                np.testing.assert_allclose(f[:, :2], x['cv_ols4'], rtol=0, atol=1e-10)
                if len(hist) != record['online_histories'] or y['stable_match'].sum() != record['stable']:
                    raise ValueError('Unmatched histories must remain on the roster')
                _, inv, count = np.unique(x['start'], return_inverse=True, return_counts=True)
                weight = 1./count[inv]
            target = study.PRIVATE/'raster_registered_outcomes'/a/scene/f'{video}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_suffix('.pending.npz')
            np.savez(pending, **y)
            pending.replace(target)
            values = f, y['target_xy'], y['target_mask'], weight
            for g in ('all', scene):
                groups[a][g].append(values)
            records[name][a] = dict(inputs=ih, corrected_outcomes=study.sha(target), metrics=readout(*values))
    metrics = {a: {g: readout(*(np.concatenate(v) for v in zip(*rows, strict=True)))
                   for g, rows in group.items()} for a, group in groups.items()}
    study.immutable(raster.OUT/'full_online_readout.json', dict(source='fresh_run', inputs='cached_verified',
        parents={str(p.relative_to(study.PUBLIC)): study.sha(p) for p in files}, code=study.sha(Path(__file__)),
        role='development diagnostic; no policy fitting, no risk gate', input_outcome_files_separate=True,
        weighting='equal over all nonempty online queries; normalized by observed CV error mass',
        records=records, arms=metrics))
    print({a: v['all'] for a, v in metrics.items()}, flush=True)


if __name__ == '__main__':
    run()
