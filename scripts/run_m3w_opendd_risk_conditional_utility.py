"""Continue OpenDD plan ranking with utility labels within the risk-passing set."""

import argparse
from collections import Counter, defaultdict
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == '__main__' and (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_equal_spend_preference as base
from src.evaluation import m3w_risk_conditional_utility as teacher


run, study, parent, np, torch, old = base.run, base.study, base.parent, base.np, base.torch, base.old
prior, save = base.prior, base.previous.saved_plan
OUT = base.OUT.parent / 'opendd_risk_conditional_utility_v1'
PRIVATE = base.PRIVATE.parent / OUT.name
STEPS = 300
CONTROL, PRIMARY = 'continued_pairwise', 'conditional_utility'
ARMS = ('frozen', 'original_pairwise', CONTROL, PRIMARY)


def register():
    key, motion_key, cells, ranks, heads, online, motion = base.register()
    p, f = (base.OUT / k for k in ('readout.json', 'head_freeze.json'))
    archived, fit = run.read(p), run.read(f)
    if (archived['registration'] != key or fit['registration'] != key or archived['checkpoint_manifest'] != run.sha(f)
            or archived['external_outcome_access'] or not archived['original_2pct'] or archived['summary']['pairwise']['dual_positive'] != 59):
        raise ValueError('Expected the verified initial pairwise heads and readout')
    files = [base.OUT / 'registration.json', p, f, OUT / 'protocol.md', Path(__file__), Path(teacher.__file__)]
    dest = OUT / 'registration.json'
    run.once(dest, run.stable(dict(inputs=[run.artifact(k) for k in files], steps=STEPS, batch_per_stream=base.BATCH,
        workers=0, threads=4, seeds=study.prior.SEEDS, primary=f'{PRIMARY}_minus_{CONTROL}',
        original_2pct=True, external_outcome_access=False, no_threshold_search=True)))
    return run.sha(dest), key, motion_key, cells, ranks, heads, online, motion, archived, fit['checkpoints']


def conditional_examples(file, identity, risk_file, d, bank, menu, cut):
    x, legal, rows, path, _ = menu
    cv, ols = bank[:2]
    move = prior.model.movement_prefix(cv, path, rows)
    names = sorted(set(d['rec']))

    def record(ordinal):
        ai, q = base.record_masks(d, names[ordinal])
        r = rows[q].copy()
        on = r >= 0
        agents = np.flatnonzero(ai)
        if not np.isin(r[on], agents).all():
            raise ValueError('Conditional pairs moved nominees between recordings')
        r[on] = np.searchsorted(agents, r[on])
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}

        def measured(plan):
            p = old.prediction(cv[ai], path[ai], r, plan)
            result = parent.score(local, cv[ai], ols[ai], p, cut)
            return bool(result['risk_pass']), min(result['vs_cv'], result['vs_ols4'])

        return ai, q, r, local, measured

    if not file.exists():
        groups = defaultdict(list)
        for v in run.read(risk_file)['entries']:
            groups[v['ordinal'], tuple(v['anchor']), tuple(v['good'])].append(v['bad'])
        out, audit = [], Counter()
        for (i, anchor, good), bad in groups.items():
            ai, q, rows0, local, measured = record(i)
            anchor, good = np.asarray(anchor), np.asarray(good)
            m, l = move[q], legal[q]
            cap = save(m, l, anchor, good)
            terms = old.prefix_terms(cv[ai], ols[ai], path[ai], local['y'], local['valid'], rows0, l, cut[0])
            expected, upper = measured(good)
            if not expected or upper <= 0.:
                raise ValueError('Missing verified donor winner')
            items = {tuple(v) for v in bad} | {tuple(v) for v in teacher.neighbors(good, m, cap, terms)}
            for k in sorted(items):
                worse = np.asarray(k)
                save(m, l, anchor, worse)
                ok, lower = measured(worse)
                if not ok or lower >= upper - 1e-6:
                    continue
                out.append(dict(ordinal=i, anchor=anchor.tolist(), good=good.tolist(), bad=worse.tolist()))
                audit['both_dual_pass'] += int(lower > 0.)
                audit['one_dual_pass'] += int(lower <= 0.)
        run.once(file, run.stable(dict(identity=identity, entries=out, audit=dict(audit, pairs=len(out), anchors=len(groups)))))
    pack = run.read(file)
    if pack['identity'] != identity or pack['audit']['pairs'] != len(pack['entries']):
        raise ValueError('Conditional donor cache changed')
    data = []
    for v in pack['entries']:
        _, q, _, _, measured = record(v['ordinal'])
        anchor, good, bad = (np.asarray(v[k]) for k in ('anchor', 'good', 'bad'))
        for plan in (good, bad):
            save(move[q], legal[q], anchor, plan)
        a, u = measured(good)
        b, v = measured(bad)
        if not a or not b or v >= u - 1e-6:
            raise ValueError('Both paired plans must pass unchanged risk and differ in utility')
        keep = (good > 0) | (bad > 0)
        data.append((x[q][keep], legal[q][keep], np.column_stack((good, bad))[keep], np.sqrt(float(good.sum()))))
    return data, pack['audit']


def batcher(risk, auxiliary):
    a, b = base.batcher(risk), base.batcher(auxiliary)

    def batch(rng):
        s, t = rng.integers(2**63 - 1, size=2)
        return a(np.random.default_rng(s)) + b(np.random.default_rng(t))

    return batch


def loss(net, x, mask, count, group, scale, x1, mask1, count1, group1, scale1):
    return .5 * (base.model.objective(net(x, mask), count, group, scale, pairwise=True)
                 + base.model.objective(net(x1, mask1), count1, group1, scale1, pairwise=True))


def evaluate(site, seed, motion_key, d, bank, menu, frozen, nets, cut, archive, diagnostic):
    cv, ols = bank[:2]
    x, legal, rows, path, _ = menu
    with torch.inference_mode():
        t, l = torch.from_numpy(x), torch.from_numpy(legal)
        f = frozen(t, l).numpy()
        scores = {k: net(t, l).numpy() for k, net in nets.items()}
    online = study.model.streaming_count(f, legal, d['recordings'], d['starts'], study.RATE)
    plans, failures = {k: online.copy() for k in ARMS}, Counter({k: 0 for k in nets})
    move = prior.model.movement_prefix(cv, path, rows)
    for i, name in enumerate(sorted(set(d['rec']))):
        _, q = base.record_masks(d, name)
        m, l, a = move[q], legal[q], online[q]
        r = run.read(prior.PRIVATE / f'{site}_{seed}_{i:04d}.json')
        if r['identity'] != dict(registration=motion_key, site=site, seed=seed, ordinal=i):
            raise ValueError('Unbound original budget')
        default = np.asarray(r['score_plan'])
        cap = save(m, l, a, default)
        if cap != r['cap'] or a.sum() != r['edits']:
            raise ValueError('Original budget changed')
        plans['frozen'][q] = default
        ix = np.arange(len(a))
        for k, s in scores.items():
            score = s[q].astype(float)
            plan, _ = prior.model.allocate(l, m, int(a.sum()), cap, score=score)
            if plan is None or score[ix, plan].sum() + 1e-8 < max(score[ix, a].sum(), score[ix, default].sum()):
                failures[k] += 1
                plan = default
            save(m, l, a, plan)
            plans[k][q] = plan
    paths = {k: old.prediction(cv, path, rows, c) for k, c in plans.items()}
    summary = {k: parent.score(d, cv, ols, p, cut) for k, p in paths.items()}
    if summary['frozen'] != archive['scores']['frozen'] or summary['original_pairwise'] != archive['scores']['pairwise']:
        raise ValueError('Matched frozen or original pairwise score changed')
    if any(s['edits'] != diagnostic['edits'] for s in summary.values()):
        raise ValueError('Matched comparison changed realized edit count')
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv - y, axis=-1)[valid].sum()
    pairs = ((PRIMARY, CONTROL), (PRIMARY, 'frozen'), (PRIMARY, 'original_pairwise'), (CONTROL, 'frozen'))
    effects = {f'{a}_minus_{b}': dict(pass_rate_pp=100. * (summary[a]['dual_positive'] - summary[b]['dual_positive']) / summary[b]['views'],
        conservative_gain_pp=float(100. * parent.transfer.paired(paths[b], paths[a], y, valid)[1].sum() / mass)) for a, b in pairs}
    return dict(scores=summary, effects=effects, unresolved=dict(failures))


def summarize(groups):
    flat = [r for rows in groups.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    fields = ('views', 'edits', 'dual_positive', 'risk_pass', 'undefined', 'unknown_edits')

    def ci(key, field):
        values = np.array([np.mean([r['effects'][key][field] for r in rows]) for rows in groups.values()])
        return dict(point_pp=float(values.mean()), ci95_pp=np.quantile(values[draw].mean(1), [.025, .975]).tolist(),
                    positive_locations=int((values > 0.).sum()))

    return dict(arms={k: {f: sum(r['scores'][k][f] for r in flat) for f in fields} for k in ARMS},
        comparisons={k: {f: ci(k, f) for f in ('pass_rate_pp', 'conservative_gain_pp')} for k in flat[0]['effects']},
        utility_pairs_across_cells=sum(r['utility_pairs'] for r in flat),
        utility_pairs_per_cell=[min(r['utility_pairs'] for r in flat), max(r['utility_pairs'] for r in flat)],
        unresolved={k: sum(r['unresolved'][k] for r in flat) for k in ARMS[1:]})


def execute(reg, prev_key, motion_key, cells, ranks, heads, online, motion, archived, fitted, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        f = parent.prior.prior.old.source.prior.PRIVATE / f'{s}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    result, checkpoints, updates = {}, {}, 0
    (PRIVATE / 'pairs').mkdir(exist_ok=True)

    def cuts(roster):
        return np.quantile(np.concatenate([sites[s]['cut_samples'] for s in roster]), [.25, .75])

    rank_net = lambda: study.prior.model.SwapValue(False)
    for held in sites:
        donors = [s for s in sites if s != held]
        result[held] = []
        for cell in cells['by_site'][held]:
            seed = cell['seed']
            init = fitted[f'{held}_{seed}_pairwise']
            if any(init['identity'][k] != v for k, v in dict(registration=prev_key, held=held, seed=seed, train=donors, mode='pairwise').items()):
                raise ValueError('Initial model changed training roles')
            ordinary, utility, hashes, audit = [], [], {}, {}
            for donor in donors:
                omitted = sorted((held, donor))
                name = f'{"_".join(omitted)}_{seed}_{donor}.json'
                risk_file = base.PRIVATE / 'pairs' / name
                if run.sha(risk_file) != init['identity']['pairs'][donor]:
                    raise ValueError('Initial donor plans changed')
                train = [s for s in sites if s not in omitted]
                meta = heads['rank_heads']['rank_' + '_'.join(omitted) + f'_{seed}']
                if meta['identity'] != dict(registration=heads['registration'], omitted=omitted, fit=train, seed=seed):
                    raise ValueError('Donor producer used a held site')
                rank = study.load(meta, rank_net, study.RANK_STEPS)
                menu = base.previous.menu_for(sites[donor], banks[donor], rank)
                identity = dict(registration=prev_key, omitted=omitted, seed=seed, donor=donor, rank=meta['artifact']['sha256'])
                cut = cuts(train)
                a, _ = base.donor_examples(risk_file, identity, sites[donor], banks[donor], menu, cut, seed)
                path = PRIVATE / 'pairs' / name
                b, audit[donor] = conditional_examples(path, dict(registration=reg, source=init['identity']['pairs'][donor]),
                    risk_file, sites[donor], banks[donor], menu, cut)
                ordinary.append(a)
                utility.append(b)
                hashes[donor] = run.sha(path)
                beat(phase='donor', held=held, seed=seed, donor=donor, pairs=len(b))
            rank = study.load(ranks['heads'][f'{held}/{seed}/additive_rank'], rank_net, study.prior.STEPS)
            menu = base.previous.menu_for(sites[held], banks[held], rank)
            width = menu[0].shape[-1]
            new_net = lambda: study.model.CountHead(width)
            start_net = lambda: study.load(init, new_net, base.STEPS)
            frozen = study.load(heads['count_heads'][f'count_{held}_{seed}_recording'], new_net, study.HEAD_STEPS)
            nets = {'original_pairwise': start_net()}
            for arm, second in ((CONTROL, ordinary), (PRIMARY, utility)):
                name = f'{held}_{seed}_{arm}'
                ident = dict(registration=reg, held=held, seed=seed, train=donors, arm=arm,
                             initial_checkpoint=init['artifact']['sha256'], risk_pairs=init['identity']['pairs'], utility_pairs=hashes)
                nets[arm], checkpoints[name], n = study.fit(ident, str(PRIVATE / name), seed, start_net,
                    STEPS, batcher(ordinary, second), loss, replay, beat)
                updates += n
            ref = next(r for r in archived['by_site'][held] if r['seed'] == seed)
            cap = next(r for r in motion['by_site'][held] if r['seed'] == seed)
            score = evaluate(held, seed, motion_key, sites[held], banks[held], menu, frozen, nets, cuts(donors), ref, cap)
            result[held].append(dict(seed=seed, utility_pairs=sum(len(v) for v in utility), donor_audit=audit, **score))
            beat(phase='cell', held=held, seed=seed, updates=updates)
    path = OUT / 'head_freeze.json'
    run.once(path, run.stable(dict(registration=reg, checkpoints=checkpoints)))
    summary = summarize(result)
    run.once(OUT / 'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(path), summary=summary,
        by_site=result, original_2pct=True, external_outcome_access=False, deployment_changed=False, offline_matching=True,
        final_fitted_updates=STEPS * len(checkpoints), submission_ready=False)))
    return updates, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**event):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **event)
            run.kin.prior.source.atomic(PRIVATE / 'heartbeat.json', row)
            if event['phase'] in ('cell', 'complete'):
                print(row, flush=True)

        updates, summary = execute(*register(), args.replay, beat)
        beat(phase='complete', updates=updates, replay=args.replay, summary=summary)


if __name__ == '__main__':
    main()
