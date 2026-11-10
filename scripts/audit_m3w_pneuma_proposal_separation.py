"""Measure frozen Athens forecast separation without reading any outcomes."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
PILOT = PUBLIC/'pneuma_prefix_pilot_v1'
LOCAL = ROOT/'data/stage_cvpr2027_experiments'/PILOT.name
OUT = PUBLIC/'pneuma_proposal_separation_v1'/'readout.json'
ARMS = ('independent', 'additive', 'same_family', 'joint')


def digest(path):
    with path.open('rb') as f:
        return sha256(f.read()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def separate(cv, offsets, radius, proposals):
    """Each paired full-grid ADE/FDE contrast is bounded by forecast distance."""
    offsets, radius = np.asarray(offsets), np.asarray(radius)
    n = len(offsets)-1
    if (cv.shape != (int(offsets[-1]), 12, 2) or offsets[0] != 0
            or np.any(np.diff(offsets) < 2) or radius.shape != (n,)
            or not np.isfinite(cv).all() or not np.all(np.isfinite(radius) & (radius > 0.))):
        raise ValueError('Invalid frozen forecast grid or edit radii')
    edits = {}
    for arm in ARMS:
        rows, paths = (proposals[f'{arm}_{suffix}'] for suffix in ('rows', 'paths'))
        if (rows.shape != (n, 2) or not np.issubdtype(rows.dtype, np.integer)
                or paths.shape != (n, 2, 12, 2) or not np.isfinite(paths).all()
                or (rows[:, 0] == rows[:, 1]).any()
                or (rows < offsets[:-1, None]).any() or (rows >= offsets[1:, None]).any()):
            raise ValueError('Invalid frozen two-recipient edit')
        delta = paths.astype('float64')-cv[rows]
        np.testing.assert_allclose(np.linalg.norm(delta, axis=-1).max(axis=-1),
                                   np.broadcast_to(radius[:, None], rows.shape), atol=1e-9, rtol=1e-9)
        edits[arm] = (rows, delta)
    joint, ref = edits['joint']
    result = {}
    for arm, (rows, delta) in edits.items():
        if arm == 'joint':
            continue
        groups, ade, fde, relative, integral = [], [], [], [], []
        for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            # Compare by target identity, not by rank in an arm's edit list.
            a = {int(i): v for i, v in zip(joint[j], ref[j], strict=True)}
            b = {int(i): v for i, v in zip(rows[j], delta[j], strict=True)}
            diff = np.stack([a.get(i, 0.)-b.get(i, 0.) for i in sorted(a.keys() | b.keys())])
            norms = np.linalg.norm(diff, axis=-1)
            i, k = np.triu_indices(len(diff), 1)
            pairs = np.linalg.norm(diff[i]-diff[k], axis=-1).mean(axis=1).sum()
            targets = hi-lo
            relative.append(float((pairs+(targets-len(diff))*norms.mean(axis=1).sum())/(targets*(targets-1)//2)))
            groups.append(a.keys() != b.keys())
            ade.append(float(norms.mean(axis=1).sum()/(hi-lo)))
            fde.append(float(norms[:, -1].sum()/(hi-lo)))
            integral.append(float(np.linalg.norm(ref[j], axis=-1).mean()-np.linalg.norm(delta[j], axis=-1).mean()))
        result[arm] = dict(different_recipients=int(sum(groups)), distinct_paths=int(np.sum(np.array(ade) > 1e-12)),
            mean_full_grid_ade_envelope=float(np.mean(ade)), mean_full_grid_fde_envelope=float(np.mean(fde)),
            mean_full_grid_relative_motion_envelope=float(np.mean(relative)),
            max_query_ade_envelope=float(np.max(ade)),
            mean_joint_minus_control_edit_path_ade=float(np.mean(integral)))
    return result


def run():
    paths = [PILOT/f'{k}_receipt.json' for k in ('prefix', 'forecast', 'matched')]
    paths += [PUBLIC/'pneuma_source_intake_v1'/'samples.json']
    a, b, c, source = map(load, paths)
    bindings = {p.name: digest(p) for p in paths}
    if (c['binding']['prefix'] != bindings['prefix_receipt.json']
            or c['binding']['forecast'] != bindings['forecast_receipt.json']
            or a['binding']['samples'] != bindings['samples.json']
            or c['future_coordinates_loaded'] or c['predictive_roles_assigned']
            or c['forecast_accuracy_scored'] or c['seeds'] != [17, 29, 43]
            or set(a['files']) != set(source['members']) or set(a['files']) != set(b['files'])):
        raise ValueError('Require frozen unscored proposals, with matching source metadata')
    by_date = defaultdict(list)
    count = 0
    for name, inputs in a['files'].items():
        stem = Path(name).stem
        f, prefix = (LOCAL/f'{stem}.{kind}.npz' for kind in ('forecast', 'prefix'))
        if (digest(f) != b['files'][name]['forecast_sha256']
                or digest(prefix) != inputs['prefix_sha256']
                or inputs['prefix_sha256'] != b['files'][name]['prefix_sha256']):
            raise ValueError('Forecast or prefix hash mismatch')
        with np.load(f, allow_pickle=False) as forecast, np.load(prefix, allow_pickle=False) as past:
            cv, offsets = forecast['cv'].astype('float64'), past['offsets']
        seeds = []
        for seed in c['seeds']:
            key = f'{stem}_{seed}'
            p = LOCAL/'matched'/f'{key}.npz'
            info = c['files'][key]
            if (digest(p) != info['proposals_sha256'] or info['identity']['name'] != name
                    or info['identity']['seed'] != seed or info['queries'] != len(offsets)-1
                    or info['identity']['prefix_sha256'] != inputs['prefix_sha256']
                    or info['identity']['forecast_sha256'] != b['files'][name]['forecast_sha256']):
                raise ValueError('Proposal packet does not match the frozen roster')
            with np.load(p, allow_pickle=False) as z:
                np.testing.assert_array_equal(offsets, z['offsets'])
                seeds.append(separate(cv, offsets, z['radius'], z))
        count += (len(offsets)-1)*len(seeds)
        by_date[source['members'][name]['day']].append((len(offsets)-1, seeds))
    if count != c['query_seed_views'] or len(by_date) != 4:
        raise ValueError('Incomplete frozen proposal bank')
    dates = {}
    for day, members in sorted(by_date.items()):
        rows = [r for _, s in members for r in s]
        stats = {}
        for arm in ARMS[:-1]:
            stats[arm] = {k: (int(sum(r[arm][k] for r in rows)) if k in ('different_recipients', 'distinct_paths')
                              else max(r[arm][k] for r in rows) if k.startswith('max_')
                              else float(np.mean([np.mean([r[arm][k] for r in s]) for _, s in members])))
                          for k in rows[0][arm]}
        dates[day] = dict(members=len(members), queries=sum(n for n, _ in members), metrics=stats)
    result = dict(result_source='fresh_frozen_forecast_geometry_no_labels', inputs=bindings, views=count,
                  dates=dates, coordinates='fixed_pilot_tangent_plane_not_accuracy_verified',
                  forecast_accuracy_scored=False, future_coordinates_loaded=False, predictive_roles_assigned=False)
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if OUT.exists():
        if OUT.read_text() != data:
            raise ValueError('Refusing to replace a differing result')
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open('x') as f:
            f.write(data)
    print(f'Verified {count} unscored forecast views across {len(dates)} dates.')


if __name__ == '__main__':
    run()
