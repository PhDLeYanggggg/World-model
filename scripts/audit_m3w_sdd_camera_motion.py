"""Measure the camera-history penalty on the same matched SDD development targets."""
from collections import defaultdict
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_sdd_camera_support as support

np = support.np
METHODS = ('cv', 'ols4', 'ols8', 'median_pair_slope')


def forecasts(history):
    last = history[:, -1]
    a, b = np.triu_indices(history.shape[1], k=1)
    pair = (history[:, b] - history[:, a]) / (b-a)[None, :, None]

    def ols(n):
        clock = np.arange(n) - (n-1)/2
        return np.einsum('t,ntd->nd', clock, history[:, -n:] - last[:, None]) / (clock @ clock)

    slopes = np.stack((last-history[:, -2], ols(4), ols(8), np.median(pair, axis=1)), axis=1)
    return last[:, None, None] + slopes[:, :, None]*np.arange(1, support.FUTURE+1)[None, None, :, None]


def report(errors, mask, non_generated, keys):
    # Compare the same track-windows and valid future steps under both inputs.
    def slice_metrics(selected):
        if not selected.any():
            return None
        values = np.where(selected[:, None, None, :], errors, 0.).sum((0, 3)) / selected.sum()
        return {source: dict(zip(METHODS, value.tolist(), strict=True))
                for source, value in zip(('camera', 'publisher'), values, strict=True)}

    complete = mask.all(1)
    cv_positive = ((errors[:, 0, 0] * mask).sum(1) > 0) & complete
    end = mask[:, -1]
    return dict(rows=len(mask), track_windows_with_label=int(mask.any(1).sum()),
                distinct_recording_tracks=len(set(keys)), complete=int(complete.sum()),
                complete_camera_cv_positive_error=int(cv_positive.sum()),
                known_step_mean_error_px=slice_metrics(mask),
                complete_ade_px=slice_metrics(mask & complete[:, None]),
                non_generated_step_mean_error_px=slice_metrics(non_generated),
                available_endpoint_fde_px=slice_metrics(mask & end[:, None] & (np.arange(mask.shape[1]) == mask.shape[1]-1)))


def run(with_aerial):
    d, reg = support.checked_registration()
    readout = support.PUBLIC / 'readout.json'
    paired = support.load(readout)
    if paired['registration'] != reg or any(support.sha(support.PUBLIC / f'{a}_freeze.json') != h
                                            for a, h in paired['freezes'].items()):
        raise ValueError('Require the frozen support readout')
    recordings = {a: {k: row[a] for k, row in paired['recordings'].items()} for a in support.ARMS}
    sources = {'primary': support.sha(readout)}
    if with_aerial:
        home = support.PUBLIC / 'aerial'
        readout = home / 'readout.json'
        a = support.load(readout)
        if a['freeze'] != support.sha(home / 'freeze.json') or a['registration'] != support.sha(home / 'registration.json'):
            raise ValueError('Aerial motion audit requires its bound support readout')
        if support.load(home / 'registration.json')['parent'] != reg:
            raise ValueError('Aerial arm must use the same camera roster')
        recordings['aerial'] = a['recordings']
        sources['aerial'] = support.sha(readout)
    out = {}
    for arm, ledger in recordings.items():
        groups = defaultdict(list)
        for s in d['sources']:
            scene, video = s['scene'], s['video']
            name = f'{scene}/{video}'
            path = support.PRIVATE / 'paired' / arm / scene / f'{video}.npz'
            _, ann = support.files(scene, video)
            if support.sha(path) != ledger[name]['paired_cache_sha256'] or support.sha(ann) != s['annotation_sha256']:
                raise ValueError('Paired input lineage changed')
            with np.load(path, allow_pickle=False) as z:
                t, people = z['start'], z['publisher_id']
                if len(t) == 0:
                    continue
                wanted = {int(a)+j for a in np.unique(t) for j in range(support.OBS)}
                labels = support.label_geometry(ann, wanted)
                clean = np.array([[labels[int(a)+j][int(i)][0] for j in range(support.OBS)]
                                  for a, i in zip(t, people, strict=True)])
                centers = lambda xyxy: .5*(xyxy[:, :, :2] + xyxy[:, :, 2:4])
                camera, publisher = (forecasts(centers(x)) for x in (z['past_xyxy_conf_class'], clean))
                if not np.allclose(camera[:, :2], z['cv_ols4'], rtol=0, atol=1e-10):
                    raise ValueError('Matched camera forecasts did not reproduce')
                err = np.linalg.norm(np.stack((camera, publisher), axis=1)-z['target_xy'][:, None, None], axis=-1)
                row = (err, z['target_mask'], z['non_generated_mask'], [(name, int(i)) for i in people])
                groups[scene].append(row)
                groups['all'].append(row)

        def aggregate(rows):
            a, b, c, names = zip(*rows, strict=True)
            return report(np.concatenate(a), np.concatenate(b), np.concatenate(c), sum(names, []))

        out[arm] = {scene: aggregate(rows) for scene, rows in groups.items()}
    output = 'motion_diagnostic_all_arms.json' if with_aerial else 'motion_diagnostic.json'
    support.immutable(support.PUBLIC / output, dict(source='fresh_run', registration=reg,
        readout_sha256=sources, implementation_sha256=support.sha(Path(__file__)), arms=out,
        unit='pixel', role='matched-track input diagnostic only; unmatched camera tracks are not evaluated here',
        approved_risk_gate_applied=False, no_model_fit=True, independent_confirmation=False, policy_changed=False))
    print({arm: d.get('all') for arm, d in out.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--with-aerial', action='store_true')
    run(cli.parse_args().with_aerial)
