"""Fit held-location angular residual heads and compare at frozen exchanges."""

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
from scripts import audit_m3w_opendd_exchange_directions as direction
from src.world_model import m3w_exchange_angular_head as model
from src.world_model.m3w_ols_replacement import rollout


run, np, torch, matched, old, base = direction.run, direction.np, direction.torch, direction.matched, direction.old, direction.base
OUT = direction.OUT.parent/'opendd_exchange_angular_head_v1'
PRIVATE = matched.exp.PRIVATE.parent/OUT.name
study = matched.exp.study
ARMS = ('no_swap', 'original', 'acceleration', 'turn', 'suffix', 'observed', 'relative')
MASKS = ('guarded', 'all_eligible')


def donor(d):
    h = d['h']
    cv, ols, _, _ = rollout(h, 12)
    take = d['valid'].any(1) & (np.linalg.norm(ols-cv, axis=-1).max(1) > 1e-10)
    cv, ols, h = (np.repeat(x[take], 3, axis=0) for x in (cv, ols, h))
    a = np.tile([.25, .5, 1.], take.sum())
    x, scale = model.features(h, cv, ols, a)
    target, valid = (np.repeat(d[k][take], 3, axis=0) for k in ('y', 'valid'))
    target = np.where(valid[..., None], target, 0.)
    anchor = cv+a[:, None, None]*(ols-cv)
    return tuple(torch.from_numpy(z.astype(np.float32)) for z in (x, cv, ols, anchor, target, scale)) + (torch.from_numpy(valid),)


def batcher(data):
    active = [z for z in data if len(z[0])]
    if not active:
        raise ValueError('No donor trajectories')

    def batch(rng):
        n = np.bincount(rng.integers(len(active), size=old.BATCH), minlength=len(active))
        idx = [rng.integers(len(s[0]), size=k) for s, k in zip(active, n, strict=True)]
        return tuple(torch.cat([s[k][i] for s, i in zip(active, idx, strict=True)]) for k in range(7))

    return batch


def evaluate(args, nets):
    site, seed, motion, d, bank, menu, width, frozen, cut = args
    x, legal, rows, path, _ = menu
    cv, ols = bank[:2]
    h = d['h']
    frac = direction.model.bridge_fraction(cv, ols, path)
    feature, _ = model.features(h, cv, ols, frac)
    dirs = direction.model.direction_bank(h, cv, path)
    pick, _ = direction.model.select(direction.model.suffix_scores(h, frac))
    pool = {'original': path, 'acceleration': dirs[:, 3], 'turn': dirs[:, 4], 'suffix': dirs[np.arange(len(h)), pick]}
    with torch.inference_mode():
        for k, net in nets.items():
            z = net(torch.from_numpy(feature)).to(torch.float64)
            pool[k] = model.rotate(z, torch.from_numpy(cv), torch.from_numpy(path)).numpy()
    rad = np.linalg.norm(path-cv, axis=-1)
    for p in pool.values():
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1), rad, atol=1e-11, rtol=1e-11)
    move = base.model.movement_prefix(cv, path, rows)
    predictions = {mask: {k: cv.copy() for k in ARMS} for mask in MASKS}
    counts = Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound original count')
        pairs = old.model.exchange_menu(a, legal[q], move[q], cap)
        p0 = old.measure(d, ai, bank, path, r, a, cut)[0]
        actions = dict.fromkeys(MASKS)
        if len(pairs):
            feat = old.model.features(x[q], legal[q], move[q], a, cap, pairs)[0]
            with torch.inference_mode():
                z = frozen['aligned'](matched.exp.design(torch.from_numpy(feat), width, 'aligned'))
            actions['guarded'], actions['all_eligible'] = old.model.rank(z), matched.matched_choice(z)
        counts.update(views=1, edits=int(a.sum()), candidates=len(pairs))
        for mask in MASKS:
            out = predictions[mask]
            out['no_swap'][ai] = p0
            j = actions[mask]
            if j is None:
                for k in pool:
                    out[k][ai] = p0
                continue
            counts[mask] += 1
            src, dst = pairs[j]
            p = old.measure(d, ai, bank, path, r, old.model.exchanged(a, src, dst), cut)[0]
            dest = r[dst, a[dst]]
            if dest < 0:
                raise ValueError('Illegal receiver')
            g = np.flatnonzero(ai)[dest]
            for k, v in pool.items():
                out[k][ai] = p
                out[k][g] = v[g]
    target, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv-target, axis=-1)[valid].sum()
    counts.update(known_steps=int(valid.sum()), endpoints=int(valid[:, -1].sum()))
    masks = {}
    for m, paths in predictions.items():
        scores, effect = {}, {}
        for k, p in paths.items():
            s, fields = direction.metric_rows(p, cv, ols, d, cut)
            if s['edits'] != counts['edits']:
                raise ValueError('Directional head changed matched budget')
            scores[k] = fields
            effect[k] = {b: float(100*direction.prior.parent.transfer.paired(paths[b], p, target, valid)[1].sum()/mass)
                         for b in ('no_swap', 'original', 'observed') if b != k}
        masks[m] = dict(arms=scores, paired=effect)
    return dict(counts=dict(counts), masks=masks)


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    counts = dict(sum((Counter(r['counts']) for r in rows), Counter()))

    def ci(fn):
        x = np.asarray([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    masks = {}
    for m in MASKS:
        arms = {k: dict(sum((Counter(r['masks'][m]['arms'][k]) for r in rows), Counter())) for k in ARMS}
        for a in arms.values():
            a.update(available_step_mean_error=a.pop('step_error')/counts['known_steps'],
                     available_endpoint_mean_error=a.pop('endpoint_error')/counts['endpoints'])
        pairs = {}
        for a, b in (('observed', 'original'), ('relative', 'original'), ('relative', 'observed'),
                     ('relative', 'no_swap'), ('observed', 'no_swap')):
            pairs[f'{a}_minus_{b}'] = dict(gain=ci(lambda r: r['masks'][m]['paired'][a][b]),
                dual=ci(lambda r: 100.*(r['masks'][m]['arms'][a]['dual']-r['masks'][m]['arms'][b]['dual'])/r['counts']['views']))
        masks[m] = dict(arms=arms, paired=pairs)
    return dict(counts=counts, masks=masks)


def execute(replay, beat):
    source = direction.OUT/'readout.json'
    out = OUT/'registration.json'
    matched.exp.register()
    loader = matched.parent.prior.prior.old
    root = loader.source.prior.PRIVATE
    sites = {s: loader.load_site(s) for s in matched.parent.source.SITES}
    inputs = [Path(__file__), Path(model.__file__), OUT/'protocol.md', source, direction.OUT/'registration.json']
    lineage = {s: run.artifact(root/f'{s}_cohort.npz') for s in sites}
    run.once(out, run.stable(dict(inputs=[run.artifact(f) for f in inputs], lineage=lineage,
        threads=4, workers=0, steps=old.STEPS, seeds=study.prior.SEEDS, original_2pct=True,
        no_held_tuning=True, external_outcome_access=False)))
    reg = run.sha(out)
    samples = {s: donor(d) for s, d in sites.items()}
    fits, groups, updates = {}, {}, 0
    for args, expected in matched.context_cells():
        site, seed = args[:2]
        train = [t for t in sites if t != site]
        batch = batcher([samples[t] for t in train])
        nets = {}
        for k, which in (('observed', 0), ('relative', 1)):
            name = f'{site}_{seed}_{k}'
            ident = dict(registration=reg, held=site, train=train, seed=seed, arm=k)

            def loss(net, x, cv, ols, p, y, scale, valid):
                return model.losses(model.rotate(net(x), cv, p), cv, ols, y, valid, scale)[which]

            nets[k], fits[name], n = study.fit(ident, str(PRIVATE/name), seed,
                lambda: model.AngularHead(samples[train[0]][0].shape[1]), old.STEPS, batch, loss, replay, beat)
            updates += n
        v = evaluate(args, nets)
        if v['counts'].get('guarded', 0) != expected['transitions']['aligned']['exchanged'] or any(
                v['masks']['guarded']['arms']['original'][a] != expected['scores']['aligned'][b]
                for a, b in (('dual', 'dual_positive'), ('risk_pass', 'risk_pass'), ('undefined', 'undefined'))):
            raise ValueError('Frozen guarded control did not reproduce')
        groups.setdefault(site, []).append(dict(seed=seed, **v))
        beat(phase='cell', held=site, seed=seed, updates=updates)
    s = summarize(groups)
    c = s['counts']
    if (c['views'], c['edits'], c['candidates'], c['guarded'], c['all_eligible']) != (606, 780, 7198, 101, 296):
        raise ValueError('Matched exchange opportunity changed')
    f = OUT/'head_freeze.json'
    run.once(f, run.stable(dict(registration=reg, checkpoints=fits)))
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(f),
        summary=s, by_site=groups, original_2pct=True, external_outcome_access=False, deployment_changed=False,
        offline_development=True, configured_updates=len(fits)*old.STEPS)))
    beat(phase='complete', updates=updates, replay=replay, summary=s)


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
