"""Fit three spline forecasters at the frozen OpenDD EqMotion budget."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch Python')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_comoving_eqmotion as control
from scripts import train_m3w_causal_velocity_spline as donor_spline
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_opendd_spline_frames as model

np, torch, run = control.np, control.torch, control.run
OUT = control.OUT.parent / 'opendd_matched_spline_v1'
PRIVATE = control.PRIVATE.parent / OUT.name
STEPS, BATCH = control.STEPS, control.BATCH
SEEDS = control.two.study.prior.SEEDS
ARM_PAIRS = dict(comoving_vs_cv=('cv', 'comoving_spline'),
    comoving_vs_eqmotion=('comoving_ego', 'comoving_spline'),
    cv_anchor_vs_ols=('ols_spline', 'cv_spline'), comoving_vs_centered=('cv_spline', 'comoving_spline'))


def sampler(data, fitting, mode):
    """Exactly the old site order, eligible-row rule, and RNG calls."""
    fit = [data[s] for s in fitting]
    elig = [np.flatnonzero(d['valid'].numpy().any(1)) for d in fit]
    if any(not len(p) for p in elig):
        raise ValueError('Require observed fit labels in each site')

    def draw(rng):
        sizes = rng.multinomial(BATCH, np.full(len(fit), 1./len(fit)))
        rows = [r[rng.integers(len(r), size=k)] for r, k in zip(elig, sizes, strict=True)]
        arrays = [torch.cat([d['inputs'][mode][j][i] for d, i in zip(fit, rows, strict=True)])
                  for j in range(6)]
        return (*arrays, *(torch.cat([d[k][i] for d, i in zip(fit, rows, strict=True)])
                           for k in ('scale', 'target', 'valid')))

    return draw


@torch.inference_mode()
def forecast(net, inputs):
    result = np.concatenate([net(*(x[i:i+512] for x in inputs), False).numpy()
                             for i in range(0, len(inputs[0]), 512)]).astype(float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite forecast')
    return result


def save(path, forecast):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(forecast):
                raise ValueError('Frozen roster changed')
            for k, p in forecast.items():
                np.testing.assert_array_equal(z[k], p)
    else:
        tmp = path.with_suffix('.pending.npz')
        np.savez(tmp, **forecast)
        tmp.replace(path)
    return run.artifact(path)


def register():
    parent_hash, _, _, _, _, source = control.register()
    fp, rp = (control.OUT / n for n in ('head_freeze.json', 'analysis.json'))
    f, r = map(run.read, (fp, rp))
    if (f['registration'] != parent_hash or r['registration'] != parent_hash
            or r['heads'] != run.sha(fp) or r['confirmation_access'] or not r['original_2pct']):
        raise ValueError('Require verified held-site EqMotion forecasts')
    s = donor_spline.OUT / 'readout.json'
    sr = run.read(s)
    if sr['independent_confirmation'] or sr['a_labels_opened']:
        raise ValueError('Require exposed-development spline result')
    p = OUT / 'registration.json'
    run.once(p, run.stable(dict(files=[run.artifact(x) for x in (
        Path(__file__), Path(model.__file__), Path(donor_spline.model.__file__), OUT/'protocol.md')],
        parents=[run.artifact(x) for x in (control.OUT/'registration.json', fp, rp, s)],
        sites=list(control.parent.source.SITES), steps=STEPS, batch=BATCH, seeds=SEEDS, workers=0, threads=4,
        matching='identical site order, valid-row sampling, optimizer, training budget and final RNG',
        primary='comoving spline minus causal CV on full held-site roster',
        original_2pct=True, confirmation_access=False, model_selection=False)))
    return run.sha(p), source, f, r


def load(source):
    data = {}
    for s in control.parent.source.SITES:
        d, records, _, eq, y, valid = control.load(s, source)
        x, scale = model.designs(d['h'])
        np.testing.assert_array_equal(scale, eq[-1].numpy())
        data[s] = dict(d=d, recording=np.repeat(records, np.unique(d['query'], return_counts=True)[1]),
            inputs={k: tuple(torch.from_numpy(v) for v in a) for k, a in x.items()},
            scale=torch.from_numpy(scale), target=y, valid=valid)
    return data


def train(data, reg, old, reference, replay, beat):
    cells, updates = [], 0
    for s in data:
        donors = [t for t in data if t != s]
        for seed in SEEDS:
            p, fits = {}, {}
            record = next(c for c in reference['by_site'][s] if c['seed'] == seed)
            check = old['heads'][f'{s}_{seed}_comoving_ego']
            prev = torch.load(run.checked(check['artifact']), weights_only=True, map_location='cpu')
            if (prev['identity'] != check['identity'] or prev['step'] != STEPS
                    or prev['identity']['fit'] != donors or prev['identity']['held'] != s
                    or prev['identity']['seed'] != seed):
                raise ValueError('Changed matched EqMotion identity')
            for arm in model.MODES:
                def cost(net, *batch):
                    return model.loss(net, batch[:6], *batch[6:])

                name = f'{s}_{seed}_{arm}'
                ident = dict(held=s, fit=donors, seed=seed, arm=arm, registration=reg)
                net, f, n = control.two.fit.fit(ident, name, seed, model.Spline,
                    STEPS, sampler(data, donors, arm), cost, replay, beat)
                fitted = torch.load(run.checked(f['artifact']), weights_only=True, map_location='cpu')
                if fitted['rng'] != prev['rng']:
                    raise ValueError('Matched EqMotion sampling RNG differs')
                p[arm] = forecast(net, data[s]['inputs'][arm])
                fits[arm] = f
                updates += n
            prediction = save(PRIVATE/f'{s}_{seed}_paths.npz', p)
            cells.append(dict(held=s, fit=donors, seed=seed, fits=fits, paths=prediction,
                eqmotion_paths=record['prediction'], eqmotion_fit=check['artifact']))
            beat(phase='frozen_cell', held=s, seed=seed, updates=updates)
    return cells, updates


def risk_summary(d, paths, cv, rec, cut):
    results = {}
    for name, p in paths.items():
        metrics = policy(p, cv, cv, d['y'], d['valid'], rec, easy_cut=cut[0], hard_cut=cut[1])
        records = list(metrics['recordings'].values())
        signed = [q['known_easy_net_degradation'] for q in records
                  if q['known_easy_net_degradation'] is not None]
        results[name] = dict(views=len(records), passes=sum(q['supported'] for q in records),
            undefined_easy=sum(q['completion_easy_risk_upper'] is None for q in records),
            worst_signed_easy_percent=100.*max(signed) if signed else None,
            switch_rate=metrics['switch_rate'])
    return results


def score(data, cells, beat):
    rows = []
    for c in cells:
        s = c['held']
        site = data[s]
        d = site['d']
        cv, ols, *_ = control.parent.prior.analytic(d)
        t = np.arange(1., 13.)[None, :, None]
        np.testing.assert_allclose(site['inputs']['ols_spline'][-1].numpy()[:, None]*t, ols, atol=1e-6)
        np.testing.assert_allclose(site['inputs']['cv_spline'][-1].numpy()[:, None]*t, cv, atol=1e-6)
        cut = np.quantile(np.concatenate([data[k]['d']['cut_samples'] for k in c['fit']]), [.25, .75])
        with (np.load(run.checked(c['paths']), allow_pickle=False) as z,
              np.load(run.checked(c['eqmotion_paths']), allow_pickle=False) as eq):
            p = {**{k: z[k] for k in model.MODES}, **{k: eq[k] for k in control.model.ARMS}}
        raw = control.raw_score(d, (cv, ols), p, cut)
        p.update(cv=cv, ols4=ols)
        mass = float(np.linalg.norm(cv-d['y'], axis=-1)[d['valid']].sum())
        diffs = {}
        for key, (l, r) in ARM_PAIRS.items():
            a, b = control.paired(p[l], p[r], d['y'], d['valid'])
            diffs[key] = dict(known=float(100.*a.sum()/mass), lower=float(100.*b.sum()/mass))
        rows.append(dict(site=s, seed=c['seed'], raw=raw, paired=diffs,
            risk=risk_summary(d, p, cv, site['recording'], cut)))
        beat(phase='held_scored', held=s, seed=c['seed'])
    return rows


def summarize(rows, data):
    sites = list(data)
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def interval(fn):
        x = np.array([np.mean([fn(r) for r in rows if r['site'] == s]) for s in sites])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()), positive_sites=int((x > 0.).sum()))

    arms = rows[0]['raw']['arms']
    raw = {k: dict(complete_ade=sum(r['raw']['arms'][k]['complete_ade_sum'] for r in rows)
                  /sum(r['raw']['complete'] for r in rows),
                  endpoint_fde=sum(r['raw']['arms'][k]['endpoint_fde_sum'] for r in rows)
                  /sum(r['raw']['endpoint'] for r in rows),
                  known={b: interval(lambda r: r['raw']['arms'][k]['known'][b]) for b in ('cv', 'ols4')},
                  lower={b: interval(lambda r: r['raw']['arms'][k]['lower'][b]) for b in ('cv', 'ols4')})
           for k in arms}
    effects = {k: {field: interval(lambda r: r['paired'][k][field]) for field in ('known', 'lower')} for k in ARM_PAIRS}
    risk = {}
    for k in arms:
        r = [x['risk'][k] for x in rows]
        vals = [x['worst_signed_easy_percent'] for x in r if x['worst_signed_easy_percent'] is not None]
        risk[k] = dict(**{f: sum(x[f] for x in r) for f in ('views', 'passes', 'undefined_easy')},
                       worst_signed_easy_percent=max(vals) if vals else None)
    status = dict(sites=len(sites), recordings=sum(len(set(d['recording'])) for d in data.values()),
        rows=sum(len(d['d']['h']) for d in data.values()),
        complete=sum(int(d['valid'].all(1).sum()) for d in data.values()),
        wholly_missing=sum(int((~d['valid'].any(1)).sum()) for d in data.values()))
    if status != dict(sites=7, recordings=202, rows=10475, complete=9823, wholly_missing=53):
        raise ValueError(f'Expected published local roster: {status}')
    return dict(roster=status, raw=raw, risk=risk, effects=effects,
                strong_direct_candidate=effects['comoving_vs_cv']['lower']['interval95'][0] > 0.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    control.two.fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', z)
            if v.get('step', STEPS) == STEPS:
                print(z, flush=True)

        reg, source, parent, reference = register()
        data = load(source)
        cells, updates = train(data, reg, parent, reference, args.replay, beat)
        fp = OUT/'freeze.json'
        run.once(fp, run.stable(dict(registration=reg, cells=cells,
            fits=len(cells)*len(model.MODES), updates=len(cells)*len(model.MODES)*STEPS, before_held_scoring=True)))
        rows = score(data, run.read(fp)['cells'], beat)
        summary = summarize(rows, data)
        run.once(OUT/'readout.json', run.stable(dict(freeze=run.sha(fp), summary=summary, rows=rows,
            original_2pct=True, confirmation_access=False, independent_confirmation=False, deployment_changed=False)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
