"""Train and score an explicit recording-risk head on fixed exchange views."""

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
from scripts import audit_m3w_opendd_exchange_attrition as audit
from src.world_model import m3w_exchange_risk_heads as model


matched, run, old, np, torch = audit.matched, audit.run, audit.before, audit.np, audit.torch
exp, study, parent, base, prefix = matched.exp, matched.exp.study, matched.parent, matched.base, matched.prefix
OUT = audit.OUT.parent/'opendd_exchange_risk_heads_v1'
PRIVATE = exp.PRIVATE.parent/OUT.name
ARMS = ('frozen', 'aligned', 'control', 'aux_transition', 'factored', 'random')


def donor_status(d, bank, menu, cut, seed, previous):
    x, legal, rows, path, _ = menu
    cv = bank[0]
    move = base.model.movement_prefix(cv, path, rows)
    rng = np.random.default_rng(seed+913)
    labels, offset = [], 0
    for name in sorted(set(d['rec'])):
        ai, q, r = old.indices(d, name, rows)
        if np.linalg.norm(cv[ai]-d['y'][ai], axis=-1)[d['valid'][ai]].sum() <= 0.:
            continue
        l, m = legal[q], move[q]
        schedules = (np.tile([0., 1., -1.], (len(l), 1)),
                     np.column_stack((np.zeros(len(l)), rng.standard_normal((len(l), 2)))))
        visited = set()
        for scores in schedules:
            a = study.model.streaming_count(scores, l, d['recordings'][q], d['starts'][q], study.RATE)
            if tuple(a) in visited:
                continue
            visited.add(tuple(a))
            cap = float(m[np.arange(len(a)), a].sum())
            pairs = old.model.exchange_menu(a, l, m, cap)
            if not len(pairs):
                continue
            if len(pairs) > old.SAMPLE:
                pairs = pairs[np.sort(rng.choice(len(pairs), size=old.SAMPLE, replace=False))]
            feat = old.model.features(x[q], l, m, a, cap, pairs)[0]
            if not np.array_equal(feat, previous[0][offset:offset+len(pairs)].numpy()):
                raise ValueError('New target changed donor design or order')
            anchor = old.measure(d, ai, bank, path, r, a, cut)[1]
            for src, dst in pairs:
                s = old.measure(d, ai, bank, path, r, old.model.exchanged(a, src, dst), cut)[1]
                if 1+s['dual_positive']-anchor['dual_positive'] != int(previous[1][offset]):
                    raise ValueError('Transition target did not reproduce')
                labels.append(audit.RISKS.index(audit.status(s)))
                offset += 1
    if offset != len(previous[0]):
        raise ValueError('Donor population changed')
    return np.asarray(labels, dtype=np.int64)


def data_for(held, donor, seed, sites, banks, maps, heads, reg, prior_reg):
    omitted = sorted((held, donor))
    roster = [s for s in sites if s not in omitted]
    meta = heads['rank_heads']['rank_'+'_'.join(omitted)+f'_{seed}']
    if meta['identity'] != dict(registration=heads['registration'], omitted=omitted, fit=roster, seed=seed):
        raise ValueError('Donor proposal producer used the donor or held location')
    name = f'{"_".join(omitted)}_{seed}_{donor}'
    p = exp.PRIVATE/'donors'/(name+'.npz')
    prev = run.read(p.with_suffix('.json'))
    ident = dict(registration=prior_reg, omitted=omitted, donor=donor, seed=seed, rank=meta['artifact']['sha256'])
    if prev['identity'] != ident or prev['cache_sha256'] != run.sha(p):
        raise ValueError('Unbound previous donor')
    with np.load(p, allow_pickle=False) as z:
        previous = tuple(torch.from_numpy(z[k]) for k in ('x', 'event', 'gain'))
    out = PRIVATE/'donors'/(name+'.json')
    link = dict(registration=reg, previous=run.sha(p.with_suffix('.json')))
    if not out.exists():
        d, bank = sites[donor], banks[donor]
        rank = study.load(meta, lambda: study.prior.model.SwapValue(False), study.RANK_STEPS)
        menu, _ = exp.augment(d, bank, prefix.menu_for(d, bank, rank), maps[donor])
        cut = np.quantile(np.concatenate([sites[t]['cut_samples'] for t in roster]), [.25, .75])
        new = donor_status(d, bank, menu, cut, seed, previous)
        run.once(out, run.stable(dict(identity=link, target=new.tolist())))
    r = run.read(out)
    if r['identity'] != link or len(r['target']) != len(previous[0]):
        raise ValueError('Unbound risk targets')
    return (*previous, torch.tensor(r['target'], dtype=torch.long)), run.sha(out)


def batcher(table):
    active = [s for s in table if len(s[0])]
    if not active:
        raise ValueError('No donor exchanges')

    def batch(rng):
        counts = np.bincount(rng.integers(len(active), size=old.BATCH), minlength=len(active))
        picks = [rng.integers(len(s[0]), size=n) for s, n in zip(active, counts, strict=True)]
        return tuple(torch.cat([s[k][i] for s, i in zip(active, picks, strict=True)]) for k in range(4))

    return batch


def evaluate(inputs, nets):
    site, seed, motion_key, d, bank, menu, width, anchors, cut = inputs
    x, legal, rows, path, _ = menu
    move = base.model.movement_prefix(bank[0], path, rows)
    totals, arms = Counter(), {k: Counter() for k in ARMS}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion_key, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound frozen count or cap')
        pairs = old.model.exchange_menu(a, legal[q], move[q], cap)
        chosen = {}
        if len(pairs):
            z = torch.from_numpy(old.model.features(x[q], legal[q], move[q], a, cap, pairs)[0])
            with torch.inference_mode():
                z = exp.design(z, width, 'aligned')
                i0 = old.model.rank(anchors['aligned'](z))
                if i0 is not None:
                    z0, z1 = nets['control'](z), nets['factored'](z)
                    chosen = dict(aligned=i0, control=matched.matched_choice(z0[:, :4]),
                                  aux_transition=matched.matched_choice(z1[:, :4]),
                                  factored=int(model.success_log_probability(z1).argmax()))
        p0 = audit.status(old.measure(d, ai, bank, path, r, a, cut)[1])
        totals.update(views=1, edits=int(a.sum()), switched=int(bool(chosen)))
        opts = ([audit.status(old.measure(d, ai, bank, path, r, old.model.exchanged(a, *p), cut)[1]) for p in pairs]
                if chosen else [])
        for k in ARMS:
            c = (Counter({r: n/len(opts) for r, n in Counter(opts).items()}) if k == 'random' and chosen
                 else Counter([opts[chosen[k]] if k in chosen else p0]))
            c['rescued'] = c['dual_pass'] if p0 != 'dual_pass' else 0
            c['destroyed'] = 1-c['dual_pass'] if p0 == 'dual_pass' else 0
            arms[k].update(c)
    return dict(**totals, arms={k: dict(v) for k, v in arms.items()})


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def compare(a, b):
        x = np.array([np.mean([100.*(r['arms'][a].get('dual_pass', 0)-r['arms'][b].get('dual_pass', 0))/r['views']
                              for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    return dict(**{k: sum(r[k] for r in rows) for k in ('views', 'edits', 'switched')},
                arms={k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in ARMS},
                paired={k: compare('factored', k) for k in ('frozen', 'control', 'aux_transition', 'aligned', 'random')})


def execute(replay, beat):
    prior, _, _, _, _, _, heads, _, _, _, assets = exp.register()
    receipt = audit.OUT/'readout.json'
    p = OUT/'registration.json'
    files = (OUT/'protocol.md', Path(__file__), Path(model.__file__), receipt, audit.OUT/'registration.json',
             exp.OUT/'head_freeze.json')
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], arms=ARMS, workers=0, threads=4,
        steps=old.STEPS, seeds=study.prior.SEEDS, no_held_tuning=True, external_outcome_access=False, original_2pct=True)))
    reg = run.sha(p)
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        f = parent.prior.prior.old.source.prior.PRIVATE/f'{s}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    maps = {s: exp.road.raster_for(s, assets[s])[0] for s in sites}
    groups, fitted, updates = {}, {}, 0
    for inputs, ref in matched.context_cells():
        held, seed = inputs[:2]
        width = inputs[6]
        donors = [t for t in sites if t != held]
        table, bindings, nets = [], {}, {}
        for t in donors:
            data, bindings[t] = data_for(held, t, seed, sites, banks, maps, heads, reg, prior)
            table.append(data)
            beat(phase='donor', held=held, seed=seed, donor=t)
        batch = batcher(table)
        for arm in ('control', 'factored'):
            name = f'{held}_{seed}_{arm}'
            ident = dict(registration=reg, held=held, train=donors, seed=seed, arm=arm, data=bindings)

            def loss(net, x, transition, gain, status):
                return model.loss(net(exp.design(x, width, 'aligned')), transition, gain, status, arm == 'factored')

            nets[arm], fitted[name], n = study.fit(ident, str(PRIVATE/name), seed,
                lambda: model.RiskRanker(4*inputs[5][0].shape[1]+9), old.STEPS, batch, loss, replay, beat)
            updates += n
        row = evaluate(inputs, nets)
        if row['switched'] != ref['transitions']['aligned']['exchanged'] or (
                row['arms']['aligned'].get('dual_pass', 0) != ref['scores']['aligned']['dual_positive']):
            raise ValueError('Original frozen choices did not reproduce')
        groups.setdefault(held, []).append(dict(seed=seed, **row))
        beat(phase='cell', held=held, seed=seed, updates=updates)
    summary = summarize(groups)
    old_summary = run.read(receipt)['summary']
    if any(summary[a] != old_summary['counts'][b] for a, b in (
            ('views', 'views'), ('edits', 'edits'), ('switched', 'switched'))) or not np.isclose(
                summary['arms']['random']['dual_pass'], old_summary['arms']['random']['dual_pass']):
        raise ValueError('Equal-spend comparison changed')
    p0 = OUT/'head_freeze.json'
    run.once(p0, run.stable(dict(registration=reg, checkpoints=fitted)))
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(p0),
        summary=summary, by_site=groups, configured_updates=len(fitted)*old.STEPS, no_held_tuning=True,
        external_outcome_access=False, original_2pct=True, offline_development=True, deployment_changed=False)))
    beat(phase='complete', updates=updates, replay=replay, summary=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    (PRIVATE/'donors').mkdir(parents=True, exist_ok=True)
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
