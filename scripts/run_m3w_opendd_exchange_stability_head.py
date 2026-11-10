"""Test background-stability supervision at the unchanged exchange budget."""

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
from scripts import audit_m3w_opendd_exchange_stability as stress
from scripts import run_m3w_opendd_exchange_risk_heads as risk
from src.world_model import m3w_exchange_stability_head as head


run, np, torch, matched, old, exp, study = risk.run, risk.np, risk.torch, risk.matched, risk.old, risk.exp, risk.study
OUT = stress.OUT.parent/'opendd_exchange_stability_head_v1'
PRIVATE = risk.PRIVATE.parent/OUT.name
ARMS = ('frozen', 'aligned', 'control', 'aux_full', 'stable', 'random')


def examples(d, bank, menu, cut, seed, batch):
    x, legal, rows, path, _ = menu
    cv = bank[0]
    move = risk.base.model.movement_prefix(cv, path, rows)
    rng = np.random.default_rng(seed+913)
    labels, offset = [], 0
    for rec in sorted(set(d['rec'])):
        ai, q, r = old.indices(d, rec, rows)
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
        c, o = cv[ai], bank[1][ai]
        if np.linalg.norm(c-local['y'], axis=-1)[local['valid']].sum() <= 0.:
            continue
        query = np.searchsorted(np.flatnonzero(q), d['query'][ai])
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
            if not np.array_equal(feat, batch[0][offset:offset+len(pairs)].numpy()):
                raise ValueError('Stability supervision changed donor design or row order')
            p0 = old.measure(d, ai, bank, path, r, a, cut)[0]
            for src, dst in pairs:
                label = -1
                if batch[3][offset] == 3:
                    p, s = old.measure(d, ai, bank, path, r, old.model.exchanged(a, src, dst), cut)
                    passed, _ = stress.stress(local, c, o, p0, p, cut, s, query, len(a), src, dst)
                    if passed['three_blocks_supported']:
                        label = int(passed['stable_dual'])
                labels.append(label)
                offset += 1
    if offset != len(batch[0]):
        raise ValueError('Stability supervision changed the donor population')
    return labels


def donor(held, s, seed, sites, banks, maps, heads, reg, risk_reg, context_reg):
    batch, parent = risk.data_for(held, s, seed, sites, banks, maps, heads, risk_reg, context_reg)
    omitted = sorted((held, s))
    p = PRIVATE/'donors'/f'{"_".join(omitted)}_{seed}_{s}.json'
    identity = dict(registration=reg, parent=parent)
    if not p.exists():
        rank = study.load(heads['rank_heads']['rank_'+'_'.join(omitted)+f'_{seed}'],
                          lambda: study.prior.model.SwapValue(False), study.RANK_STEPS)
        d, bank = sites[s], banks[s]
        menu, _ = exp.augment(d, bank, risk.prefix.menu_for(d, bank, rank), maps[s])
        cut = np.quantile(np.concatenate([sites[t]['cut_samples'] for t in sites if t not in omitted]), [.25, .75])
        run.once(p, run.stable(dict(identity=identity, stable=examples(d, bank, menu, cut, seed, batch))))
    r = run.read(p)
    if r['identity'] != identity or len(r['stable']) != len(batch[0]):
        raise ValueError('Unbound stability target')
    return (*batch, torch.tensor(r['stable'], dtype=torch.long)), run.sha(p)


def batcher(table):
    active = [a for a in table if len(a[0])]
    if not active:
        raise ValueError('No training exchanges')

    def batch(rng):
        n = np.bincount(rng.integers(len(active), size=old.BATCH), minlength=len(active))
        pick = [rng.integers(len(s[0]), size=m) for s, m in zip(active, n, strict=True)]
        return tuple(torch.cat([s[k][i] for s, i in zip(active, pick, strict=True)]) for k in range(5))

    return batch


def evaluate(args, nets):
    s, seed, motion, d, bank, menu, width, frozen, cut = args
    x, legal, rows, path, _ = menu
    move = risk.base.model.movement_prefix(bank[0], path, rows)
    tally, arms = Counter(), {a: Counter() for a in ARMS}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(risk.base.PRIVATE/f'{s}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=s, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound edit count')
        pairs = old.model.exchange_menu(a, legal[q], move[q], cap)
        choices = {}
        if len(pairs):
            z = torch.from_numpy(old.model.features(x[q], legal[q], move[q], a, cap, pairs)[0])
            with torch.inference_mode():
                z = exp.design(z, width, 'aligned')
                aligned = old.model.rank(frozen['aligned'](z))
                if aligned is not None:
                    b, u = nets['control'](z), nets['stability'](z)
                    choices = dict(aligned=aligned, control=head.rank(b, False), aux_full=head.rank(u, False),
                                   stable=head.rank(u, True))
        p0, anchor = old.measure(d, ai, bank, path, r, a, cut)
        tally.update(views=1, edits=int(a.sum()), switched=int(bool(choices)))
        full, stable = [], []
        if choices:
            local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
            query = np.searchsorted(np.flatnonzero(q), d['query'][ai])
            for src, dst in pairs:
                p, z = old.measure(d, ai, bank, path, r, old.model.exchanged(a, src, dst), cut)
                full.append(z['dual_positive'])
                stable.append(int(not anchor['dual_positive'] and z['dual_positive'] and stress.stress(
                    local, bank[0][ai], bank[1][ai], p0, p, cut, z, query, len(a), src, dst)[0]['stable_dual']))
        for k in ARMS:
            ok, st = anchor['dual_positive'], 0
            if k in choices:
                ok, st = full[choices[k]], stable[choices[k]]
            elif k == 'random' and choices:
                ok, st = float(np.mean(full)), float(np.mean(stable))
            arms[k].update(dual_pass=ok, stable_rescue=st, rescued=ok if not anchor['dual_positive'] else 0,
                           destroyed=1-ok if anchor['dual_positive'] else 0)
    return dict(**tally, arms={k: dict(v) for k, v in arms.items()})


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def compare(b, field):
        x = np.asarray([np.mean([100.*(r['arms']['stable'][field]-r['arms'][b][field])/r['views']
                                  for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    return dict(**{k: sum(r[k] for r in rows) for k in ('views', 'edits', 'switched')},
                arms={k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in ARMS},
                stable_minus={k: {f: compare(k, f) for f in ('dual_pass', 'stable_rescue')} for k in ARMS if k != 'stable'},
                donor_stability_labels=sum(r['donor_stability_labels'] for r in rows),
                donor_stable_positives=sum(r['donor_stable_positives'] for r in rows))


def execute(replay, beat):
    context_reg, _, _, _, _, _, heads, _, _, _, assets = exp.register()
    r = risk.OUT/'registration.json'
    risk_reg = run.sha(r)
    check = run.read(risk.OUT/'readout.json')
    if check['registration'] != risk_reg or check['external_outcome_access'] or not check['original_2pct']:
        raise ValueError('Require the frozen risk experiment')
    p = OUT/'registration.json'
    files = [OUT/'protocol.md', Path(__file__), Path(head.__file__), r, risk.OUT/'readout.json',
             stress.OUT/'registration.json', stress.OUT/'readout.json']
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], steps=old.STEPS, seeds=study.prior.SEEDS,
        arms=ARMS, workers=0, threads=4, original_2pct=True, external_outcome_access=False)))
    reg = run.sha(p)
    sites = {s: risk.parent.prior.prior.old.load_site(s) for s in risk.parent.source.SITES}
    for s, d in sites.items():
        f = risk.parent.prior.prior.old.source.prior.PRIVATE/f'{s}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
    banks = {s: risk.parent.prior.analytic(d) for s, d in sites.items()}
    maps = {s: exp.road.raster_for(s, assets[s])[0] for s in sites}
    groups, fits, updates = {}, {}, 0
    for args, ref in matched.context_cells():
        s, seed = args[:2]
        train = [t for t in sites if t != s]
        table, ids, nets = [], {}, {}
        for t in train:
            a, ids[t] = donor(s, t, seed, sites, banks, maps, heads, reg, risk_reg, context_reg)
            table.append(a)
            beat(phase='donor', held=s, seed=seed, donor=t)
        for k in ('control', 'stability'):
            name = f'{s}_{seed}_{k}'

            def loss(net, x, transition, gain, status, stable):
                z = net(exp.design(x, args[6], 'aligned'))
                return head.loss(z, transition, gain, status, stable, k == 'stability')

            identity = dict(registration=reg, held=s, train=train, seed=seed, arm=k, data=ids)
            nets[k], fits[name], n = study.fit(identity, str(PRIVATE/name), seed,
                lambda: head.StabilityRanker(4*args[5][0].shape[1]+9), old.STEPS, batcher(table), loss, replay, beat)
            updates += n
        row = evaluate(args, nets)
        if row['arms']['aligned']['dual_pass'] != ref['scores']['aligned']['dual_positive'] or (
                row['switched'] != ref['transitions']['aligned']['exchanged']):
            raise ValueError('Aligned reference did not reproduce')
        groups.setdefault(s, []).append(dict(seed=seed, **row,
            donor_stability_labels=sum(int((a[4] >= 0).sum()) for a in table),
            donor_stable_positives=sum(int((a[4] == 1).sum()) for a in table)))
        beat(phase='cell', held=s, seed=seed, updates=updates)
    result = summarize(groups)
    ref = check['summary']
    if any(result[k] != ref[k] for k in ('views', 'edits', 'switched')) or not np.isclose(
            result['arms']['random']['dual_pass'], ref['arms']['random']['dual_pass'], atol=1e-12, rtol=0.):
        raise ValueError('Changed matched opportunity or chance')
    f = OUT/'head_freeze.json'
    run.once(f, run.stable(dict(registration=reg, checkpoints=fits)))
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(f), by_site=groups,
        summary=result, configured_updates=len(fits)*old.STEPS, original_2pct=True, external_outcome_access=False,
        offline_development=True, deployment_changed=False)))
    beat(phase='complete', updates=updates, replay=replay, summary=result)


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
