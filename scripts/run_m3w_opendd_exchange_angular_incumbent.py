"""Fit direct incumbent-relative and donor-tail angular objectives."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == '__main__' and (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_angular_same_recipients as same
from src.world_model import m3w_exchange_angular_incumbent as model


old, fit, run, np, torch = same.old, same.fit, same.run, same.np, same.torch
OUT = fit.OUT.parent/'opendd_exchange_angular_incumbent_v1'
PRIVATE = fit.PRIVATE.parent/OUT.name
GROUP_SIZE = 24
ARMS = ('original', *model.ARMS)
CONTRASTS = tuple((a, 'original') for a in model.ARMS)+(
    ('incumbent_mean', 'dual_reference'), ('incumbent_tail', 'dual_reference'), ('incumbent_tail', 'incumbent_mean'))


def batches(data):
    if any(not len(a[0]) for a in data):
        raise ValueError('Each donor must have training trajectories')

    def sample(rng):
        picks = [rng.integers(len(a[0]), size=GROUP_SIZE) for a in data]
        return tuple(torch.cat([a[k][i] for a, i in zip(data, picks, strict=True)]) for k in range(7))

    return sample


def evaluate(args, nets):
    site, seed, motion, d, bank, menu, _, _, cut = args
    _, _, rows, path, _ = menu
    cv, ols = bank[:2]
    frac = fit.direction.model.bridge_fraction(cv, ols, path)
    x, _ = fit.model.features(d['h'], cv, ols, frac)
    paths = {'original': path}
    with torch.inference_mode():
        for k, net in nets.items():
            z = net(torch.from_numpy(x)).double()
            paths[k] = fit.model.rotate(z, torch.from_numpy(cv), torch.from_numpy(path)).numpy()
    radius = np.linalg.norm(path-cv, axis=-1)
    for p in paths.values():
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1), radius, atol=1e-11, rtol=1e-11)
    pred = {k: cv.copy() for k in ARMS}
    counts = Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, _, r = old.indices(d, rec, rows)
        f = run.read(same.base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a = np.asarray(f['score_plan'])
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound original recipients')
        keep = r[np.arange(2) < a[:, None]]
        if (keep < 0).any() or len(keep) != len(set(keep)):
            raise ValueError('Invalid original recipients')
        g = np.flatnonzero(ai)[keep]
        for k, p in paths.items():
            pred[k][g] = p[g]
        counts.update(views=1, edits=int(a.sum()))
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    counts.update(known_steps=int(valid.sum()), endpoints=int(valid[:, -1].sum()))
    actual = np.linalg.norm(pred['original']-cv, axis=-1).max(1) > 0.
    metrics, tail, gain = {}, {}, {}
    known = valid.any(1)
    for k, p in pred.items():
        np.testing.assert_array_equal(np.linalg.norm(p-cv, axis=-1).max(1) > 0., actual)
        s, z = fit.direction.metric_rows(p, cv, ols, d, cut)
        if s['edits'] != counts['edits']:
            raise ValueError('Direction changed original budget')
        metrics[k] = z
        errors = fit.direction.prior.stability.capacity.policy(p, cv, cv, y, valid, d['rec'],
            easy_cut=cut[0], hard_cut=cut[1])
        signed = [r['known_easy_net_degradation'] for r in errors['recordings'].values()
                  if r['known_easy_net_degradation'] is not None]
        err = np.linalg.norm(p-y, axis=-1)
        values = np.where(valid, err, 0.).sum(1)/valid.sum(1).clip(min=1)
        rec = d['rec'][known]
        rows = values[known]
        tail[k] = dict(available_row_p95=float(np.quantile(rows, .95)),
                       worst_recording_mean=float(max(rows[rec == n].mean() for n in set(rec))),
                       worst_known_easy_degradation_pp=100.*max(signed) if signed else None)
    for a, b in CONTRASTS:
        g, low = fit.direction.prior.parent.transfer.paired(pred[b], pred[a], y, valid)
        gain[f'{a}_minus_{b}'] = dict(observed_pp=float(100.*g.sum()/mass), lower_pp=float(100.*low.sum()/mass))
    if len({z['undefined'] for z in metrics.values()}) != 1:
        raise ValueError('Fixed recipients changed reference support')
    return dict(counts=dict(counts), arms=metrics, tails=tail, gain=gain)


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    counts = dict(sum((Counter(r['counts']) for r in rows), Counter()))

    def ci(fn):
        x = np.asarray([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    arms = {k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in ARMS}
    for a in arms.values():
        a.update(available_step_mean_error=a.pop('step_error')/counts['known_steps'],
                 available_endpoint_mean_error=a.pop('endpoint_error')/counts['endpoints'])
    paired = {}
    for a, b in CONTRASTS:
        k = f'{a}_minus_{b}'
        paired[k] = {f: ci(lambda r: r['gain'][k][f]) for f in ('observed_pp', 'lower_pp')}
        paired[k]['dual_rate_pp'] = ci(lambda r: 100.*(r['arms'][a]['dual']-r['arms'][b]['dual'])/r['counts']['views'])
        means = {s: float(np.mean([r['gain'][k]['lower_pp'] for r in g])) for s, g in groups.items()}
        paired[k].update(positive_locations=sum(x > 0. for x in means.values()), worst_location_pp=min(means.values()))
    return dict(counts=counts, arms=arms, paired=paired)


def execute(replay, beat):
    fit.matched.exp.register()
    prior = same.OUT/'readout.json'
    inputs = [Path(__file__), Path(model.__file__), OUT/'protocol.md', prior, same.OUT/'registration.json',
              fit.OUT/'registration.json', fit.OUT/'readout.json']
    loader = fit.matched.parent.prior.prior.old
    sites = {s: loader.load_site(s) for s in fit.matched.parent.source.SITES}
    inputs.extend(loader.source.prior.PRIVATE/f'{s}_cohort.npz' for s in sites)
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in inputs], arms=model.ARMS, steps=old.STEPS,
        group_size=GROUP_SIZE, original_2pct=True, threads=4, workers=0, seeds=fit.study.prior.SEEDS,
        keeps_incumbent_recipients=True, no_held_tuning=True, external_outcome_access=False)))
    reg = run.sha(p)
    samples = {s: fit.donor(d) for s, d in sites.items()}
    groups, checkpoints, updates = {}, {}, 0
    for args, ref in fit.matched.context_cells():
        site, seed = args[:2]
        donors = [s for s in sites if s != site]
        nets, batch = {}, batches([samples[s] for s in donors])
        for k in model.ARMS:
            name = f'{site}_{seed}_{k}'
            ident = dict(registration=reg, held=site, train=donors, seed=seed, arm=k)

            def loss(net, x, cv, ols, anchor, y, scale, valid):
                pred = fit.model.rotate(net(x), cv, anchor)
                return model.objective(pred, cv, ols, anchor, y, valid, scale, GROUP_SIZE, k)

            nets[k], checkpoints[name], n = fit.study.fit(ident, str(PRIVATE/name), seed,
                lambda: fit.model.AngularHead(samples[donors[0]][0].shape[1]), old.STEPS, batch, loss, replay, beat)
            updates += n
        row = evaluate(args, nets)
        if any(row['arms']['original'][a] != ref['scores']['frozen'][b]
               for a, b in (('dual', 'dual_positive'), ('risk_pass', 'risk_pass'), ('undefined', 'undefined'))):
            raise ValueError('Original held cohort did not reproduce')
        groups.setdefault(site, []).append(dict(seed=seed, **row))
        beat(phase='cell', held=site, seed=seed, updates=updates)
    summary = summarize(groups)
    p0 = run.read(prior)['summary']
    if (summary['counts'] != p0['counts'] or any(not np.isclose(z, p0['arms']['original'][k], atol=1e-12, rtol=1e-12)
            for k, z in summary['arms']['original'].items())):
        raise ValueError('Changed primary incumbent comparison')
    f = OUT/'head_freeze.json'
    run.once(f, run.stable(dict(registration=reg, checkpoints=checkpoints)))
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(f), summary=summary,
        by_site=groups, original_2pct=True, external_outcome_access=False, offline_development=True,
        configured_updates=len(checkpoints)*old.STEPS, deployment_changed=False)))
    beat(phase='complete', updates=updates, replay=replay, summary=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**event):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **event)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            if event['phase'] in ('cell', 'complete'):
                print(row, flush=True)

        execute(args.replay, beat)


if __name__ == '__main__':
    main()
