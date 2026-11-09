"""Fit and score EqMotion-core K=1 on the frozen OpenDD vehicle cohort."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_vehicle_observed_geometry as geometry
from src.world_model import m3w_eqmotion_adapter as author, m3w_vehicle_eqmotion as model

vehicle, np, prior = geometry.source, geometry.np, geometry.prior
torch = vehicle.torch
OUT = vehicle.OUT.parent / 'opendd_vehicle_eqmotion_v1'
LOCAL = vehicle.LOCAL.parent / OUT.name
ARMS = model.ARMS
SEEDS = prior.SEEDS
SITES = vehicle.SITES


def read(path):
    return json.loads(path.read_text())


def beat(phase, **kw):
    msg = dict(pid=os.getpid(), phase=phase, **kw)
    prior.write_json(LOCAL / 'heartbeat.json', msg)
    print(msg, flush=True)


def register():
    key, _ = vehicle.register()
    frozen = vehicle.OUT / 'freeze.json'
    p = vehicle.OUT / 'cohorts.json'
    g = geometry.OUT / 'manifest.json'
    baseline = read(frozen)
    geo = read(g)
    if (baseline['registration'] != key or baseline['songdo_read']
            or len(baseline['checkpoints']) != 42 or not geo['past_coordinates_only']
            or not geo['no_labels_exported'] or set(geo['sites']) != set(SITES)):
        raise ValueError('Require the frozen vehicle model and past-only aligned geometry')
    reg = OUT / 'registration.json'
    _, upstream = author.verified_source()
    files = (Path(__file__), Path(model.__file__), Path(model.eq.__file__), Path(vehicle.__file__),
             Path(vehicle.model.__file__), p, g, frozen, OUT / 'protocol.md')
    prior.write_json(reg, dict(files={str(f.relative_to(prior.ROOT)): prior.sha(f) for f in files},
        source_key=key, architecture=model.eq.ARCH, author=upstream, arms=ARMS, seeds=SEEDS, sites=SITES,
        updates=vehicle.STEPS, checkpoint_every=vehicle.SAVE, batch=vehicle.BATCH, threads=4, workers=0,
        future_targets='training and held readout only', external_outcomes_opened=False,
        model_selection='fixed final step, no held selection'), immutable=True)
    return prior.sha(reg), key, baseline, geo


def load(key, geometry_manifest):
    data = vehicle.load(key)
    cohorts = read(vehicle.OUT / 'cohorts.json')['sites']
    for site, d in data.items():
        ref = geometry_manifest['sites'][site]
        path = geometry.LOCAL / f'{site}.npz'
        if prior.sha(path) != ref['sha256'] or ref['cohort_sha256'] != cohorts[site]['sha256']:
            raise ValueError('Observed geometry is misaligned')
        with np.load(path, allow_pickle=False) as z:
            x = model.inputs(d['history'], z['observed_current_offsets'], d['query_counts'])
        cv, ols = vehicle.analytic(d['history'])
        d['cv'], d['ols4'] = cv, ols
        d['eq'] = (*x, torch.from_numpy(cv.astype(np.float32)), torch.from_numpy(d['scale'].astype(np.float32)),
                   torch.from_numpy(d['future']), torch.from_numpy(d['valid']))
    return data


def identity(key, held, seed, arm):
    return dict(registration=key, held=held, donors=[s for s in SITES if s != held], seed=seed, arm=arm)


def fit(key, data, held, seed, arm):
    meta = identity(key, held, seed, arm)
    donors = meta['donors']
    name = f'{held}_{seed}_{arm}'
    path = LOCAL / f'{name}.pt'
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, SITES.index(held)])
    net = model.eq.Forecast(arm)
    optim = torch.optim.AdamW(net.parameters(), lr=.001)
    start, began = 0, monotonic()
    if path.exists():
        state = torch.load(path, weights_only=True, map_location='cpu')
        if state['identity'] != meta or not 0 <= state['step'] <= vehicle.STEPS:
            raise ValueError('Cannot resume a different fit')
        net.load_state_dict(state['weights'])
        optim.load_state_dict(state['optim'])
        rng.bit_generator.state, start = state['rng'], state['step']
    for step in range(start+1, vehicle.STEPS+1):
        pick = rng.integers(len(donors), size=vehicle.BATCH)
        batches = []
        for i, site in enumerate(donors):
            t = data[site]['eq']
            row = rng.integers(len(t[0]), size=int((pick == i).sum()))
            batches.append(tuple(z[row] for z in t))
        optim.zero_grad()
        value = model.objective(net, tuple(torch.cat(z) for z in zip(*batches, strict=True)))
        if not torch.isfinite(value):
            raise ValueError(f'Nonfinite training loss: {name}/{step}')
        value.backward()
        optim.step()
        if step % vehicle.SAVE == 0 or step == vehicle.STEPS:
            temp = path.with_suffix('.pending')
            torch.save(dict(identity=meta, step=step, weights=net.state_dict(), optim=optim.state_dict(),
                            rng=rng.bit_generator.state), temp)
            temp.replace(path)
            beat('fit', name=name, step=step, loss=float(value.detach()), seconds=round(monotonic()-began, 2))
    return dict(identity=meta, sha256=prior.sha(path), step=vehicle.STEPS)


@torch.inference_mode()
def predict_eq(path, ref, meta, data):
    if prior.sha(path) != ref['sha256'] or ref['identity'] != meta:
        raise ValueError('Frozen checkpoint does not belong to the held fold')
    state = torch.load(path, weights_only=True, map_location='cpu')
    if state['step'] != vehicle.STEPS or state['identity'] != meta:
        raise ValueError('Require the fixed final checkpoint')
    net = model.eq.Forecast(meta['arm']).eval()
    net.load_state_dict(state['weights'])
    return torch.cat([model.bounded(net(a, b, c), cv, s) for a, b, c, cv, s in
        zip(*(t.split(128) for t in data['eq'][:5]), strict=True)]).numpy().astype(float)


@torch.inference_mode()
def predict_incumbent(site, seed, d, refs):
    name = f'{site}_{seed}_completion'
    path = vehicle.LOCAL / f'{name}.pt'
    r = refs['checkpoints'][name]
    state = torch.load(path, weights_only=True, map_location='cpu')
    if (prior.sha(path) != r['sha256'] or state['identity'] != r['identity']
            or state['step'] != vehicle.STEPS or r['identity']['donors'] != [s for s in SITES if s != site]):
        raise ValueError('Incumbent weights or source roles changed')
    net = vehicle.model.motion.network(17).eval()
    net.load_state_dict(state['weights'])
    x, cv = d['tensors'][:2]
    out = torch.cat([vehicle.model.predict(net, a, b) for a, b in
                     zip(x.split(4096), cv.split(4096), strict=True)]).numpy()
    return (out*d['scale'][:, None, None]) @ d['rotation'].transpose(0, 2, 1)


def score(key, data, incumbent):
    freeze = OUT / 'freeze.json'
    f = read(freeze)
    if f['registration'] != key or len(f['checkpoints']) != 42:
        raise ValueError('All fixed-step models must freeze before held scoring')
    prior_results = read(vehicle.OUT / 'readout.json')
    if prior_results['freeze'] != prior.sha(vehicle.OUT / 'freeze.json'):
        raise ValueError('Archived vehicle scores do not bind the checkpoints')
    easy_pool = {}
    for s, d in data.items():
        complete = d['valid'].all(1)
        e = np.linalg.norm(d['cv'][complete]-d['future'][complete], axis=-1).mean(axis=1)
        easy_pool[s] = e[e > 0]
    binding = dict(freeze=prior.sha(freeze), incumbent_readout=prior.sha(vehicle.OUT / 'readout.json'))
    rows = {}
    for s, d in data.items():
        cv, ols, y, mask, w = (d[k] for k in ('cv', 'ols4', 'future', 'valid', 'weight'))
        cut = np.quantile(np.concatenate([easy_pool[k] for k in SITES if k != s]), [.25, .75])
        mass = float(w @ np.where(mask, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        for seed in SEEDS:
            cell = f'{s}_{seed}'
            cp = LOCAL / f'{cell}_readout.json'
            if cp.exists():
                row = read(cp)
                if row['binding'] != binding:
                    raise ValueError('Resume readout belongs to another freeze')
            else:
                p = {a: predict_eq(LOCAL / f'{cell}_{a}.pt', f['checkpoints'][f'{cell}_{a}'],
                                    identity(key, s, seed, a), d) for a in ARMS}
                p.update(cv=cv, ols4=ols, completion=predict_incumbent(s, seed, d, incumbent))
                metrics = {a: vehicle.metric(v, cv, ols, d, cut) for a, v in p.items()}
                expected = prior_results['sites'][s]['seeds'][str(seed)]['arms']['completion']
                np.testing.assert_allclose(metrics['completion']['cv_gain_known_lower_pp'],
                                           expected['cv_gain_known_lower_pp'], atol=1e-9)
                def contrast(a, b):
                    return [float(100.*(w @ z.sum(1))/mass) for z in vehicle.paired(p[b], p[a], y, mask)]
                comparisons = {a+'_minus_completion': contrast(a, 'completion') for a in ARMS}
                comparisons['comoving_peers_minus_comoving_ego'] = contrast(*ARMS)
                row = dict(binding=binding, site=s, seed=seed, metrics=metrics, comparisons=comparisons)
                prior.write_json(cp, row, immutable=True)
            rows[cell] = row
            beat('score', cell=cell, completed=len(rows))
    draws = np.random.default_rng(3183).integers(len(SITES), size=(3000, len(SITES)))
    def interval(get):
        x = np.array([np.mean([get(rows[f'{s}_{seed}']) for seed in SEEDS]) for s in SITES])
        return dict(mean=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(), sites=x.tolist())
    summaries = {a: {k: interval(lambda r: r['metrics'][a][k]) for k in
        ('known_ade_native', 'fde_available_native', 'easy_gain_percent', 'p95_ade_native')} for a in (*ARMS, 'completion', 'cv', 'ols4')}
    for a, item in summaries.items():
        for ref in ('cv', 'ols'):
            item[ref+'_gain'] = {metric: interval(lambda r: r['metrics'][a][ref+'_gain_known_lower_pp'][i])
                                 for i, metric in enumerate(('known', 'lower'))}
        item['risk_passes'] = sum(r['metrics'][a]['risk_passes'] for r in rows.values())
    effects = {a: {k: interval(lambda r: r['comparisons'][a][i]) for i, k in enumerate(('known', 'lower'))}
               for a in next(iter(rows.values()))['comparisons']}
    prior.write_json(OUT / 'readout.json', dict(binding=binding, metrics=summaries, paired=effects, by_site_seed=rows,
        result_source='fresh_run', original_2pct_unchanged=True, external_outcomes_opened=False,
        independent_confirmation=False, deployment_changed=False), immutable=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg, key, incumbent, geo = register()
        data = load(key, geo)
        if args.phase == 'fit':
            weights = {f'{s}_{seed}_{a}': fit(reg, data, s, seed, a) for s in SITES for seed in SEEDS for a in ARMS}
            prior.write_json(OUT / 'freeze.json', dict(registration=reg, checkpoints=weights,
                updates=len(weights)*vehicle.STEPS, external_outcomes_opened=False), immutable=True)
        else:
            score(reg, data, incumbent)
