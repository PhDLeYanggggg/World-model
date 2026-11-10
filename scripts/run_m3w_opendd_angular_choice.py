"""Learn which fixed-angle residual to use, including no rotation."""

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
from scripts import run_m3w_opendd_exchange_angular_incumbent as prior
from src.evaluation import m3w_angular_incumbent_oracle as geometry
from src.world_model import m3w_angular_choice as model


fit, run, np, torch, old = prior.fit, prior.run, prior.np, prior.torch, prior.old
OUT = fit.OUT.parent/'opendd_angular_choice_v1'
PRIVATE = fit.PRIVATE.parent/OUT.name
ANGLES = np.deg2rad((0, *model.DEGREES))
ARMS = ('original', 'donor_constant', 'direct', 'factor', 'factor_at_direct_mask')
PAIRS = tuple((k, 'original') for k in ARMS[1:])+(('factor', 'direct'), ('factor_at_direct_mask', 'direct'))


def donor_average(table):
    totals = []
    for _, cv, _, path, y, scale, valid in table:
        a, b = model.targets(cv, path, y, valid, scale)
        totals.append((a-b).mean(0))
    s = torch.stack(totals).mean(0)
    return int(torch.cat((s.new_zeros(1), s)).argmax())


def evaluate(args, nets, constant):
    site, seed, motion, d, bank, menu, _, _, cut = args
    _, _, rows, path, _ = menu
    cv, ols = bank[:2]
    h = d['h']
    frac = fit.direction.model.bridge_fraction(cv, ols, path)
    feat, _ = fit.model.features(h, cv, ols, frac)
    with torch.inference_mode():
        x = torch.from_numpy(feat)
        direct, factor = nets['direct'](x), nets['factor'](x)
        a, b = model.choose(direct, False).numpy(), model.choose(factor, True).numpy()
        c = np.where(a > 0, model.choose(factor, True, force=True).numpy(), 0)
    picks = dict(original=np.zeros(len(path), dtype=int), donor_constant=np.full(len(path), constant),
                 direct=a, factor=b, factor_at_direct_mask=c)
    variants = {k: np.where((idx != 0)[:, None, None], geometry.rotate(cv, path, ANGLES[idx][:, None]), path)
                for k, idx in picks.items()}
    p0 = np.linalg.norm(path-cv, axis=-1)
    for p in variants.values():
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1), p0, atol=1e-11, rtol=1e-11)
    counts = Counter()
    plans = {k: cv.copy() for k in ARMS}
    used = {k: Counter() for k in ARMS}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, _, r = old.indices(d, rec, rows)
        v = run.read(prior.same.base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a = np.asarray(v['score_plan'])
        if v['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != v['edits']:
            raise ValueError('Unbound fixed recipients')
        keep = r[np.arange(2) < a[:, None]]
        if (keep < 0).any() or len(keep) != len(set(keep)):
            raise ValueError('Illegal incumbent recipients')
        take = np.flatnonzero(ai)[keep]
        counts.update(views=1, edits=int(a.sum()))
        for k in ARMS:
            plans[k][take] = variants[k][take]
            used[k].update(rotated_edits=int((picks[k][take] != 0).sum()),
                           rotated_views=int((picks[k][take] != 0).any()))
    if used['direct'] != used['factor_at_direct_mask']:
        raise ValueError('Same-rotation control lost its paired mask')
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    edit = np.linalg.norm(plans['original']-cv, axis=-1).max(1) > 0.
    counts.update(known_steps=int(valid.sum()), endpoints=int(valid[:, -1].sum()))
    scores = {}
    for k, p in plans.items():
        np.testing.assert_array_equal(np.linalg.norm(p-cv, axis=-1).max(1) > 0., edit)
        score, values = fit.direction.metric_rows(p, cv, ols, d, cut)
        if score['edits'] != counts['edits']:
            raise ValueError('Rotation changed edit count')
        scores[k] = dict(values, **used[k])
    if len({v['undefined'] for v in scores.values()}) != 1:
        raise ValueError('Rotation changed reference support')
    gain = {f'{a}_minus_{b}': float(100.*fit.direction.prior.parent.transfer.paired(
        plans[b], plans[a], y, valid)[1].sum()/mass) for a, b in PAIRS}
    return dict(counts=dict(counts), arms=scores, lower_pp=gain, donor_constant_index=constant)


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    count = dict(sum((Counter(r['counts']) for r in rows), Counter()))
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(fn):
        v = np.asarray([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(v.mean()), ci95_pp=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                    positive_locations=int((v > 0.).sum()), worst_location_pp=float(v.min()))

    arms = {k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in ARMS}
    for a in arms.values():
        a.setdefault('rotated_edits', 0)
        a.setdefault('rotated_views', 0)
        a.update(available_step_mean_error=a.pop('step_error')/count['known_steps'],
                 available_endpoint_mean_error=a.pop('endpoint_error')/count['endpoints'])
    p = {f'{a}_minus_{b}': dict(lower=ci(lambda r: r['lower_pp'][f'{a}_minus_{b}']),
        dual=ci(lambda r: 100.*(r['arms'][a]['dual']-r['arms'][b]['dual'])/r['counts']['views'])) for a, b in PAIRS}
    return dict(counts=count, arms=arms, paired=p)


def execute(replay, beat):
    fit.matched.exp.register()
    r = prior.OUT/'readout.json'
    oracle = fit.OUT.parent/'opendd_angular_incumbent_oracle_v1/readout.json'
    files = [Path(__file__), Path(model.__file__), Path(geometry.__file__), OUT/'protocol.md', r,
             prior.OUT/'registration.json', oracle]
    loader = fit.matched.parent.prior.prior.old
    sites = {s: loader.load_site(s) for s in fit.matched.parent.source.SITES}
    files.extend(loader.source.prior.PRIVATE/f'{s}_cohort.npz' for s in sites)
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], angles=(0, *model.DEGREES),
        steps=old.STEPS, group_size=prior.GROUP_SIZE, threads=4, workers=0, seeds=fit.study.prior.SEEDS,
        original_2pct=True, no_held_tuning=True, external_outcome_access=False)))
    reg = run.sha(p)
    samples = {s: fit.donor(d) for s, d in sites.items()}
    groups, checkpoints, updates = {}, {}, 0
    for args, ref in fit.matched.context_cells():
        s, seed = args[:2]
        donors = [t for t in sites if t != s]
        table = [samples[t] for t in donors]
        batch, constant = prior.batches(table), donor_average(table)
        nets = {}
        for k in ('direct', 'factor'):
            name = f'{s}_{seed}_{k}'
            ident = dict(registration=reg, held=s, train=donors, seed=seed, arm=k)

            def loss(net, x, cv, ols, anchor, y, scale, valid):
                observed, debit = model.targets(cv, anchor, y, valid, scale)
                return model.loss(net(x), observed, debit, k == 'factor')

            nets[k], checkpoints[name], n = fit.study.fit(ident, str(PRIVATE/name), seed,
                lambda: model.ChoiceHead(table[0][0].shape[1]), old.STEPS, batch, loss, replay, beat)
            updates += n
        row = evaluate(args, nets, constant)
        if any(row['arms']['original'][a] != ref['scores']['frozen'][b]
               for a, b in (('dual','dual_positive'), ('risk_pass','risk_pass'), ('undefined','undefined'))):
            raise ValueError('Fixed original recipient output changed')
        groups.setdefault(s, []).append(dict(seed=seed, **row))
        beat(phase='cell', held=s, seed=seed, updates=updates)
    result = summarize(groups)
    prev = run.read(r)['summary']
    if result['counts'] != prev['counts'] or any(not np.isclose(result['arms']['original'][k], z)
            for k, z in prev['arms']['original'].items()):
        raise ValueError('Evaluation roster changed')
    f = OUT/'head_freeze.json'
    run.once(f, run.stable(dict(registration=reg, checkpoints=checkpoints)))
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(f),
        summary=result, by_site=groups, original_2pct=True, external_outcome_access=False,
        configured_updates=len(checkpoints)*old.STEPS, deployment_changed=False)))
    beat(phase='complete', updates=updates, replay=replay, summary=result)


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
