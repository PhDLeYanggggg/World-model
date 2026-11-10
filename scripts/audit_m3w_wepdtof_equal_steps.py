"""Score equal-step movement controls on already scored WEPDTOF predictions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_wepdtof_full_grid as archived
from src.evaluation import m3w_frozen_step_control as motion
from src.evaluation import m3w_sparse_pair_bounds as relative

base, frozen, np, study = archived.base, archived.frozen, archived.np, archived.study
OUT = archived.OUT.with_name('wepdtof_equal_step_v1')
CONTROLS = ('fixed_independent', 'fixed_uniform', 'fixed_no_pair')


def receipt(x):
    h = sha256(np.ascontiguousarray(x).view(np.uint8))
    return dict(sha256=h.hexdigest(), shape=list(x.shape), dtype=str(x.dtype))


def load_geometry():
    scores = json.loads(frozen.checked(study.OUT/'external_readout.json', frozen.READOUT))
    freeze = json.loads(frozen.checked(study.OUT/'augmented_freeze.json', frozen.FREEZE))
    if (scores['forecast_freeze'] != frozen.FREEZE or not scores['external_scoring_completed']
            or scores['external_tuning'] or scores['dronecrowd_confirmation_access']
            or not scores['original_2pct'] or not freeze['outcomes_unopened'] or not freeze['original_2pct']):
        raise ValueError('Require original frozen forecasts and completed one-shot results')
    reg = study.OUT/'evaluation_registration.json'
    frozen.checked(reg, freeze['evaluation_registration'])
    camera, windows = study.frozen_camera()
    rows = {r['clip']: r for r in scores['rows']}
    if (len(rows) != 15 or len(freeze['rows']) != 15 or set(rows) != set(windows)
            or set(rows) != {r['clip'] for r in freeze['rows']}
            or len({r['source'] for r in rows.values()}) != 13 or study.checked(freeze['camera']) != camera):
        raise ValueError('Original external roster changed')
    packets, profiles = [], []
    totals = {arm: Counter() for arm in CONTROLS}
    geometry = {arm: [] for arm in CONTROLS}
    for item in freeze['rows']:
        name = item['clip']
        old = rows[name]
        h, q, ids = study.camera_arrays(windows[name])
        if (old['source'] != item['source'] or old['online'] != len(q)
                or item['online'] != len(q) or old['freeze'] != frozen.FREEZE):
            raise ValueError('Original row binding changed')
        pairs, proof = {}, {}
        with np.load(study.checked(item['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(q, z['q'])
            np.testing.assert_array_equal(ids, z['ids'])
            for seed in frozen.SEEDS:
                cv, joint = (z[f'{seed}_{k}'] for k in ('cv', 'fixed_joint'))
                a, b = {}, {}
                for arm in CONTROLS:
                    p = motion.build(cv, z[f'{seed}_{arm}'], joint, q)
                    edits = old['seeds'][str(seed)]['arms'].get('fixed_joint', {}).get('risk', {}).get('selected', 0)
                    if p['counts']['original_edits'] != edits:
                        raise ValueError('Frozen count does not reproduce the original readout')
                    totals[arm].update(p['counts'])
                    totals[arm]['identical_edited_views'] += int(edits > 0 and np.array_equal(*p['pairs']['original']))
                    retained = p['counts']['eligible_queries']-p['counts']['collapsed_queries']
                    if retained:
                        geometry[arm].append((retained, p['peak_retained'], p['path_retained']))
                    a[arm] = p
                    b[arm] = dict(**{k: p[k] for k in ('counts', 'peak_retained', 'path_retained')},
                                  kept=receipt(p['keep']), paths={k: receipt(v) for k, v in p['pairs'].items()})
                pairs[str(seed)] = (cv, a)
                proof[str(seed)] = b
        packets.append((old, h, q, ids, pairs))
        profiles.append(dict(artifact=item['artifact']['sha256'], rows=len(q), seeds=proof))
    if any(v['original_edits'] != 366 for v in totals.values()):
        raise ValueError('Expected the original full edit budget')
    source = dict(forecast_freeze=frozen.FREEZE, original_readout=frozen.READOUT,
                  driver=base.sha(Path(__file__)), math=base.sha(Path(motion.__file__)),
                  matching=base.sha(Path(motion.match.__code__.co_filename)), protocol=base.sha(OUT/'protocol.md'))
    retention = {a: {name: float(np.average(np.array(z)[:, i], weights=np.array(z)[:, 0]))
                    for i, name in enumerate(('mean_peak_retained', 'mean_integrated_retained'), 1)}
                 for a, z in geometry.items()}
    binding = dict(source=source, packets=profiles, totals={a: dict(c) for a, c in totals.items()}, retention=retention,
                   clips=15, sources=13, seeds=list(frozen.SEEDS), future_access=False, original_2pct=True)
    return binding, packets, windows


def weights(q):
    _, inv, n = np.unique(q, return_inverse=True, return_counts=True)
    return 1./n[inv]


def utility(pair, y, mask, w, mass):
    control, joint = pair
    g, low = study.paired(control, joint, y, mask)
    observed, lower = 100.*np.array([w @ g.sum(1), w @ low.sum(1)])/mass
    return dict(observed=float(observed), lower=float(lower), upper=float(2*observed-lower))


def score(cv, p, q, y, valid, old, mass_full):
    n = p['counts']['original_edits']
    pair = p['pairs']['original']
    estimable = n > 0 and mass_full > 0.
    expected = old if old is not None else [None, None]
    if estimable:
        a = utility(pair, y, valid, weights(q), mass_full)
        np.testing.assert_allclose((a['observed'], a['lower']), expected, atol=1e-12, rtol=1e-12)
    elif expected != [None, None]:
        raise ValueError('Original utility support changed')
    keep = p['keep']
    if not estimable or not p['counts']['retained_edits']:
        return None
    q, y, valid, cv = (x[keep] for x in (q, y, valid, cv))
    w = weights(q)
    mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    if mass <= 0.:
        return None
    out = {}
    groups = [np.flatnonzero(q == i) for i in np.unique(q)]
    groups = [i for i in groups if len(i) >= 2]
    for name, p in p['pairs'].items():
        a = utility(p[:, keep], y, valid, w, mass)
        ref, joint = p[:, keep]
        z = [relative.compare(joint[i], ref[i], y[i], valid[i]) for i in groups]
        available = [v['available'] for v in z if v['available'] is not None]
        a.update(relative_observed=float(np.mean(available)) if available else None,
                 relative_lower=float(np.mean([v['lower'] for v in z])) if z else None,
                 relative_upper=float(np.mean([v['upper'] for v in z])) if z else None)
        out[name] = a
    return out


def aggregate(by_source):
    def ci(v):
        a = np.asarray(v)
        if not len(a):
            return dict(point=None, ci95=None, sources=0)
        rng = np.random.default_rng(82317)
        return dict(point=float(a.mean()), ci95=np.quantile(rng.choice(a, size=(3000, len(a))).mean(1),
                    [.025, .975]).tolist(), sources=len(a))

    out, data, changes = {}, {}, {}
    fields = ('observed', 'lower', 'upper', 'relative_observed', 'relative_lower', 'relative_upper')
    for arm in CONTROLS:
        variants, source_values = {}, {}
        for variant in motion.VARIANTS:
            vv, d = {}, {}
            for f in fields:
                values = [[v[arm][variant][f] if v[arm] is not None else None for v in r]
                          for _, r in sorted(by_source.items())]
                d[f] = [float(np.mean(v)) if all(x is not None for x in v) else None for v in values]
                vv[f] = ci([x for x in d[f] if x is not None])
            variants[variant] = vv
            source_values[variant] = d
        for field in fields:
            if len({tuple(x is not None for x in r[field]) for r in source_values.values()}) != 1:
                raise ValueError('Variants must have a common source roster')
        out[arm] = variants
        data[arm] = source_values
        changes[arm] = {f: ci([a-b for a, b in zip(source_values['matched_steps'][f], source_values['original'][f],
                                                   strict=True) if a is not None]) for f in fields}
    return out, data, changes


def run(phase):
    binding, packets, windows = load_geometry()
    path = OUT/'geometry_freeze.json'
    if phase == 'freeze':
        base.immutable(path, binding)
        print(binding['totals'], flush=True)
        return
    if not path.is_file() or study.read(path) != binding:
        raise ValueError('Reproduce the frozen equal-step geometry before reading archived targets')
    by_source = defaultdict(list)
    for old, h, q, ids, pairs in packets:
        clip = old['clip']
        y, valid, matched, histories = study.outcomes(windows[clip], h, q, ids, study.references(clip))
        if (int(matched.sum()), int(valid.all(1).sum()), int((~valid.any(1)).sum()), histories) != tuple(
                old[k] for k in ('matched', 'full', 'missing', 'reference_histories')):
            raise ValueError('Archived WEPDTOF outcome support changed')
        for seed, (cv, pp) in pairs.items():
            m = float(weights(q) @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
            o = old['seeds'][seed]
            archived.replay(m, o['reference_mass'])
            r = {k: score(cv, p, q, y, valid, o['comparisons'].get(k), m) for k, p in pp.items()}
            by_source[old['source']].append(r)
    result, source_values, changes = aggregate(by_source)
    if len(by_source) != 13 or sum(map(len, by_source.values())) != 45:
        raise ValueError('Original scored views or sources changed')
    out = dict(result_source='fresh_retrospective_equal_step_sensitivity',
               freeze_sha256=base.sha(path), clips=15, total_sources=13, views=45,
               counts=binding['totals'], retention=binding['retention'], by_comparator=result,
               source_values=source_values, equal_steps_minus_original=changes, original_2pct=True,
               previously_scored_outcomes_reused=True, original_release_changed=False,
               new_external_targets_opened=False, new_fit=False, independent_confirmation=False)
    base.immutable(OUT/'readout.json', out)
    print(result['fixed_independent'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    run(p.parse_args().phase)
