"""Matched simple vehicle forecasts on the existing OpenDD development roster."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_eqmotion as eq
from src.world_model import m3w_vehicle_kinematic_blend as model

vehicle, prior, np, torch = eq.vehicle, eq.prior, eq.np, eq.torch
OUT = vehicle.OUT.parent / 'opendd_kinematic_blend_v1'
LOCAL = vehicle.LOCAL.parent / OUT.name
ARMS = ('cv', 'ols4', *model.NAMES, 'blend', 'incumbent')


def read(path):
    return json.loads(path.read_text())


def beat(**kw):
    prior.write_json(LOCAL / 'heartbeat.json', dict(pid=os.getpid(), **kw))
    print(kw, flush=True)


def register():
    k, _ = vehicle.register()
    f = vehicle.OUT/'freeze.json'
    ref = read(f)
    if ref['registration'] != k or ref['songdo_read'] or len(ref['checkpoints']) != 42:
        raise ValueError('Expected the existing vehicle checkpoints')
    p = OUT/'registration.json'
    files = (Path(__file__), Path(model.__file__), Path(eq.__file__), Path(vehicle.__file__),
             vehicle.OUT/'cohorts.json', OUT/'protocol.md', f)
    prior.write_json(p, dict(files={str(x.relative_to(prior.ROOT)): prior.sha(x) for x in files},
        cohort_registration=k, arms=ARMS, sites=vehicle.SITES, seeds=prior.SEEDS,
        sampled_rows_per_seed=vehicle.STEPS*vehicle.BATCH,
        original_2pct_unchanged=True, external_outcomes_read=False), immutable=True)
    return prior.sha(p), k, ref


def prepare(key):
    data = vehicle.load(key)
    for d in data.values():
        cv, ols = vehicle.analytic(d['history'])
        d.update(cv=cv, ols4=ols, basis=model.paths(d['history'], cv, d['scale']))
    return data


def draw_rows(data, donors, seed, held):
    rng = np.random.default_rng([seed, vehicle.SITES.index(held)])
    rows = {s: [] for s in donors}
    # Preserve the incumbent's interleaving of site and actor RNG draws.
    for _ in range(vehicle.STEPS):
        which = rng.integers(len(donors), size=vehicle.BATCH)
        for i, site in enumerate(donors):
            rows[site].append(rng.integers(len(data[site]['cv']), size=int((which == i).sum())))
    return {s: np.concatenate(x) for s, x in rows.items()}


def fit(reg, data):
    cells = {}
    for held in vehicle.SITES:
        donors = [s for s in vehicle.SITES if s != held]
        for seed in prior.SEEDS:
            k = f'{held}_{seed}'
            checkpoint = LOCAL / f'{k}_fit.json'
            binding = dict(registration=reg, held=held, seed=seed, donors=donors)
            if not checkpoint.exists():
                rows = draw_rows(data, donors, seed, held)
                x = [np.concatenate([data[s][key][rows[s]] for s in donors], axis=0)
                     for key in ('basis', 'cv', 'future', 'valid', 'scale')]
                solution = model.fit(*x)
                if solution['donor_loss'] > solution['cv_donor_loss']+1e-10:
                    raise ValueError('Convex solution must not lose to feasible CV on training')
                prior.write_json(checkpoint, dict(binding=binding, **solution), immutable=True)
            cells[k] = read(checkpoint)
            if cells[k]['binding'] != binding:
                raise ValueError('Fit checkpoint does not bind the held fold')
            beat(phase='fit', fold=k, complete=len(cells))
    prior.write_json(OUT/'freeze.json', dict(registration=reg, cells=cells, held_scoring_started=False), immutable=True)


def score(reg, data, incumbent):
    p = OUT/'freeze.json'
    frozen = read(p)
    if frozen['registration'] != reg or len(frozen['cells']) != 21:
        raise ValueError('All blend weights must freeze before held scoring')
    old = read(vehicle.OUT/'readout.json')
    if old['freeze'] != prior.sha(vehicle.OUT/'freeze.json'):
        raise ValueError('Archived vehicle scores do not bind the checkpoints')
    rows = {}
    errors = {}
    for s, d in data.items():
        complete = d['valid'].all(1)
        x = np.linalg.norm(d['cv'][complete]-d['future'][complete], axis=-1).mean(1)
        errors[s] = x[x > 0.]
    for s, d in data.items():
        cv, ols, y, mask, w = (d[k] for k in ('cv', 'ols4', 'future', 'valid', 'weight'))
        mass = w @ np.where(mask, np.linalg.norm(cv-y, axis=-1), 0.).sum(1)
        cut = np.quantile(np.concatenate([errors[k] for k in vehicle.SITES if k != s]), [.25, .75])
        p0 = dict(cv=cv, ols4=ols, **{a: model.predict(d['basis'], cv, d['scale'], v)
                                      for a, v in zip(model.NAMES, np.eye(len(model.NAMES)), strict=True)})
        cached = {a: vehicle.metric(x, cv, ols, d, cut) for a, x in p0.items()}
        for seed in prior.SEEDS:
            key = f'{s}_{seed}'
            r = frozen['cells'][key]
            if r['binding']['held'] != s or set(r['binding']['donors']) != set(vehicle.SITES)-{s}:
                raise ValueError('Held source entered the blend')
            p = dict(**p0, blend=model.predict(d['basis'], cv, d['scale'], r['weights']),
                     incumbent=eq.predict_incumbent(s, seed, d, incumbent))
            scores = dict(**cached, **{a: vehicle.metric(p[a], cv, ols, d, cut) for a in ('blend', 'incumbent')})
            expected = old['sites'][s]['seeds'][str(seed)]['arms']['completion']
            np.testing.assert_allclose(scores['incumbent']['cv_gain_known_lower_pp'], expected['cv_gain_known_lower_pp'],
                                       atol=1e-9)
            contrasts = {a: [float(100.*(w @ x.sum(1))/mass) for x in vehicle.paired(p['incumbent'], p[a], y, mask)]
                         for a in (*model.NAMES, 'blend')}
            rows[key] = dict(site=s, seed=seed, metrics=scores, paired_vs_incumbent=contrasts)
            beat(phase='score', fold=key, complete=len(rows))
    draws = np.random.default_rng(3183).integers(len(vehicle.SITES), size=(3000, len(vehicle.SITES)))
    def interval(get):
        x = np.array([np.mean([get(rows[f'{s}_{seed}']) for seed in prior.SEEDS]) for s in vehicle.SITES])
        return dict(mean=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(), sites=x.tolist())
    summary = {a: {k: interval(lambda r: r['metrics'][a][k]) for k in
               ('known_ade_native', 'fde_available_native', 'easy_gain_percent', 'p95_ade_native')} for a in ARMS}
    for a, item in summary.items():
        item['cv_gain'] = {k: interval(lambda r: r['metrics'][a]['cv_gain_known_lower_pp'][i])
                           for i, k in enumerate(('known', 'lower'))}
        item['risk_passes'] = sum(r['metrics'][a]['risk_passes'] for r in rows.values())
    contrasts = {a: {k: interval(lambda r: r['paired_vs_incumbent'][a][i]) for i, k in enumerate(('known', 'lower'))}
                 for a in (*model.NAMES, 'blend')}
    prior.write_json(OUT/'readout.json', dict(freeze_sha256=prior.sha(OUT/'freeze.json'), result_source='fresh_run',
        summary=summary, paired_vs_incumbent=contrasts, by_location_seed=rows,
        original_2pct_unchanged=True, external_outcomes_read=False, independent_confirmation=False,
        deployment_changed=False), immutable=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    a = parser.parse_args()
    LOCAL.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg, key, incumbent = register()
        data = prepare(key)
        if a.phase == 'fit':
            fit(reg, data)
        else:
            score(reg, data, incumbent)
