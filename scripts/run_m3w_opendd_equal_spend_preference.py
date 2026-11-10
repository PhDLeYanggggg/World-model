"""Learn recording-level preferences on paired plans at the same realized budget."""

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
from scripts import run_m3w_opendd_prefix_value as previous
from src.world_model import m3w_equal_spend_preference as model


prior, old, run, study, parent, np, torch = (previous.prior, previous.old, previous.run, previous.study,
    previous.parent, previous.np, previous.torch)
OUT = previous.OUT.parent / 'opendd_equal_spend_preference_v1'
PRIVATE = previous.PRIVATE.parent / OUT.name
STEPS, BATCH = 500, 32
NEW = ('pointwise', 'pairwise')
ARMS = ('frozen', *NEW)


def register():
    fit_key, motion_key, cells, ranks, heads, online, motion = previous.register()
    p = previous.OUT / 'readout.json'
    fitted = run.read(p)
    if (fitted['registration'] != fit_key or fitted['confirmation_access'] or not fitted['original_2pct']
            or fitted['summary']['matched']['frozen']['dual_positive'] != 60):
        raise ValueError('Expected the completed and unchanged prefix-value experiment')
    files = [previous.OUT / 'registration.json', p, prior.OUT / 'readout.json', OUT / 'protocol.md',
             Path(__file__), Path(model.__file__)]
    path = OUT / 'registration.json'
    run.once(path, run.stable(dict(inputs=[run.artifact(f) for f in files], seeds=study.prior.SEEDS,
        steps=STEPS, batch=BATCH, threads=4, workers=0, forecast_steps=[8, 12], original_2pct=True,
        primary='pairwise', external_outcome_access=False, deployment_changed=False, no_threshold_search=True)))
    return run.sha(path), motion_key, cells, ranks, heads, online, motion


def record_masks(d, name):
    agent, query = d['rec'] == name, d['recordings'] == name
    if (not np.array_equal(np.unique(d['query'][agent]), np.flatnonzero(query))
            or (np.diff(d['starts'][query]) <= 0).any()):
        raise ValueError('Nonchronological or misaligned recording')
    return agent, query


def donor_examples(file, identity, d, bank, menu, cut, seed):
    x, legal, rows, path, _ = menu
    cv, ols = bank[:2]
    movement = prior.model.movement_prefix(cv, path, rows)
    labels = ('y', 'valid', 'rec')
    rng = np.random.default_rng(seed + 412)
    if not file.exists():
        entries, audit = [], Counter()
        for i, name in enumerate(sorted(set(d['rec']))):
            ai, qi = record_masks(d, name)
            local = {k: d[k][ai] for k in labels}
            l, m = legal[qi], movement[qi]
            r = rows[qi].copy()
            on = r >= 0
            here = np.flatnonzero(ai)
            if not np.isin(r[on], here).all():
                raise ValueError('Nominated donor moved between recordings')
            r[on] = np.searchsorted(here, r[on])
            early = np.tile([0., 1., -1.], (len(l), 1))
            draw = np.column_stack((np.zeros(len(l)), rng.standard_normal((len(l), 2))))
            negative = np.column_stack((np.zeros(len(l)), rng.standard_normal((len(l), 2))))
            q = np.arange(len(l))
            g, risk, easy, _, offset, easy_cap, mass = old.prefix_terms(
                cv[ai], ols[ai], path[ai], local['y'], local['valid'], r, l, cut[0])
            seen, seen_anchors = set(), set()

            def passes(plan):
                p = old.prediction(cv[ai], path[ai], r, plan)
                return bool(parent.score(local, cv[ai], ols[ai], p, cut)['dual_positive'])

            for score in (early, draw):
                anchor = study.model.streaming_count(score, l, d['recordings'][qi], d['starts'][qi], study.RATE)
                if tuple(anchor) in seen_anchors:
                    continue
                seen_anchors.add(tuple(anchor))
                total, cap = int(anchor.sum()), float(m[q, anchor].sum())
                if not total:
                    continue
                audit['nonzero_anchor'] += 1
                good, status = prior.model.allocate(l, m, total, cap, terms=(g, risk, easy, offset, easy_cap, mass))
                if good is None or not passes(good):
                    audit['no_verified_witness'] += 1
                    audit['unresolved_witness'] += int(status.startswith('unresolved') or good is not None)
                    continue
                audit['verified_witness'] += 1
                previous.saved_plan(m, l, anchor, good)
                candidates = [anchor, *(prior.model.allocate(l, m, total, cap, score=rank)[0]
                                         for rank in (-m, negative))]
                for bad in candidates:
                    if bad is None:
                        audit['unresolved_control'] += 1
                        continue
                    key = tuple(good), tuple(bad)
                    if key in seen:
                        continue
                    seen.add(key)
                    previous.saved_plan(m, l, anchor, bad)
                    if passes(bad):
                        continue
                    entries.append(dict(ordinal=i, anchor=anchor.tolist(), good=good.tolist(), bad=bad.tolist()))
        run.once(file, run.stable(dict(identity=identity, entries=entries, audit=dict(audit, pairs=len(entries)))))
    pack = run.read(file)
    if pack['identity'] != identity or pack['audit']['pairs'] != len(pack['entries']):
        raise ValueError('Donor pairs changed fitting roles')
    options, names = [], sorted(set(d['rec']))
    for v in pack['entries']:
        name = names[v['ordinal']]
        ai, q = record_masks(d, name)
        anchor, good, bad = (np.asarray(v[k]) for k in ('anchor', 'good', 'bad'))
        for plan in (good, bad):
            previous.saved_plan(movement[q], legal[q], anchor, plan)
        r = rows[q].copy()
        on = r >= 0
        r[on] = np.searchsorted(np.flatnonzero(ai), r[on])
        for count, ok in ((good, True), (bad, False)):
            p = old.prediction(cv[ai], path[ai], r, count)
            if bool(parent.score({k: d[k][ai] for k in labels}, cv[ai], ols[ai], p, cut)['dual_positive']) != ok:
                raise ValueError('Donor pair failed the original scorer')
        use = (good > 0) | (bad > 0)
        options.append((x[q][use], legal[q][use], np.column_stack((good, bad))[use], np.sqrt(float(good.sum()))))
    return options, pack['audit']


def batcher(data):
    src = [d for d in data if d]
    if not src:
        raise ValueError('No verified same-budget donor pairs')

    def batch(rng):
        out = [src[i][rng.integers(len(src[i]))] for i in rng.integers(len(src), size=BATCH)]
        x, legal, options, scale = zip(*out, strict=True)
        group = np.repeat(np.arange(BATCH), [len(z) for z in x])
        return tuple(torch.from_numpy(z) for z in (np.concatenate(x), np.concatenate(legal), np.concatenate(options),
                                                    group, np.asarray(scale, dtype=np.float32)))

    return batch


def evaluate(site, seed, key, d, bank, menu, frozen, nets, cut, original, diagnostic):
    cv, ols = bank[:2]
    x, legal, rows, path, _ = menu
    with torch.inference_mode():
        x, mask = torch.from_numpy(x), torch.from_numpy(legal)
        frozen_score = frozen(x, mask).numpy()
        scores = {k: v(x, mask).numpy() for k, v in nets.items()}
    online = study.model.streaming_count(frozen_score, legal, d['recordings'], d['starts'], study.RATE)
    if parent.score(d, cv, ols, old.prediction(cv, path, rows, online), cut) != original:
        raise ValueError('Original online scores did not reproduce')
    plans = {k: online.copy() for k in ARMS}
    movement = prior.model.movement_prefix(cv, path, rows)
    failures = Counter({k: 0 for k in NEW})
    for i, name in enumerate(sorted(set(d['rec']))):
        _, q = record_masks(d, name)
        m, l, anchor = movement[q], legal[q], online[q]
        receipt = run.read(prior.PRIVATE / f'{site}_{seed}_{i:04d}.json')
        if receipt['identity'] != dict(registration=key, site=site, seed=seed, ordinal=i):
            raise ValueError('Unbound archived allocation')
        base = np.asarray(receipt['score_plan'])
        cap = previous.saved_plan(m, l, anchor, base)
        if cap != receipt['cap'] or anchor.sum() != receipt['edits']:
            raise ValueError('Frozen edit budget changed')
        plans['frozen'][q] = base
        ix = np.arange(len(anchor))
        for k in NEW:
            score = scores[k][q].astype(float)
            plan, _ = prior.model.allocate(l, m, int(anchor.sum()), cap, score=score)
            # Use the same comparator on an unresolved allocation, not a different policy.
            if plan is None or score[ix, plan].sum() + 1e-8 < max(score[ix, anchor].sum(), score[ix, base].sum()):
                failures[k] += 1
                plan = base
            previous.saved_plan(m, l, anchor, plan)
            plans[k][q] = plan
    paths = {k: old.prediction(cv, path, rows, c) for k, c in plans.items()}
    summary = {k: parent.score(d, cv, ols, p, cut) for k, p in paths.items()}
    if any(summary[k]['edits'] != diagnostic['edits'] for k in ARMS):
        raise ValueError('Learned plans changed the edit count')
    for a, b in (('dual_positive', 'score_plan'), ('risk_pass', 'score_risk'), ('undefined', 'score_undefined')):
        if summary['frozen'][a] != diagnostic[b]:
            raise ValueError('Archived matched comparator did not reproduce')
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv - y, axis=-1)[valid].sum()
    direct = {k: float(100. * parent.transfer.paired(paths['frozen'], paths[k], y, valid)[1].sum() / mass) for k in NEW}
    return dict(scores=summary, paired_lower_pp=direct, unresolved=dict(failures))


def summarize(result):
    fields = ('views', 'edits', 'dual_positive', 'risk_pass', 'undefined', 'unknown_edits')
    rows = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(result), size=(3000, len(result)))

    def paired(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    positive_locations=int((x > 0.).sum()))

    out = {k: {f: sum(r['scores'][k][f] for r in rows) for f in fields} for k in ARMS}
    for k in NEW:
        out[k]['pass_rate_minus_frozen'] = paired(lambda r: 100. * (
            r['scores'][k]['dual_positive'] - r['scores']['frozen']['dual_positive']) / r['scores']['frozen']['views'])
        out[k]['conservative_gain_minus_frozen'] = paired(lambda r: r['paired_lower_pp'][k])
        out[k]['unresolved'] = sum(r['unresolved'][k] for r in rows)
    out['training_pairs_across_cells'] = sum(r['pairs'] for r in rows)
    out['training_pairs_per_cell'] = [min(r['pairs'] for r in rows), max(r['pairs'] for r in rows)]
    return out


def execute(reg, key, cells, ranks, heads, online, motion, replay, beat):
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
            source, bindings, checks = [], {}, {}
            for donor in donors:
                omitted = sorted((held, donor))
                fit = [s for s in sites if s not in omitted]
                meta = heads['rank_heads']['rank_' + '_'.join(omitted) + f'_{seed}']
                if meta['identity'] != dict(registration=heads['registration'], omitted=omitted, fit=fit, seed=seed):
                    raise ValueError('Donor ranker must exclude both scoring and donor locations')
                rank = study.load(meta, rank_net, study.RANK_STEPS)
                m = previous.menu_for(sites[donor], banks[donor], rank)
                # The same two-location exclusion and donor determine the same teaching table.
                path = PRIVATE / 'pairs' / f'{"_".join(omitted)}_{seed}_{donor}.json'
                identity = dict(registration=reg, omitted=omitted, seed=seed, donor=donor, rank=meta['artifact']['sha256'])
                examples, audit = donor_examples(path, identity, sites[donor], banks[donor], m, cuts(fit), seed)
                source.append(examples)
                checks[donor] = audit
                bindings[donor] = run.sha(path)
                beat(phase='donor', held=held, seed=seed, donor=donor, pairs=len(examples))
            rank = study.load(ranks['heads'][f'{held}/{seed}/additive_rank'], rank_net, study.prior.STEPS)
            menu = previous.menu_for(sites[held], banks[held], rank)
            width = menu[0].shape[-1]
            frozen = study.load(heads['count_heads'][f'count_{held}_{seed}_recording'],
                                lambda: study.model.CountHead(width), study.HEAD_STEPS)
            batch, nets = batcher(source), {}
            for k in NEW:
                name = f'{held}_{seed}_{k}'
                ident = dict(registration=reg, held=held, seed=seed, train=donors, mode=k, pairs=bindings)

                def loss(net, x, mask, choices, group, scale, pairwise=k == 'pairwise'):
                    return model.objective(net(x, mask), choices, group, scale, pairwise=pairwise)

                nets[k], checkpoints[name], n = study.fit(ident, str(PRIVATE / name), seed,
                    lambda: study.model.CountHead(width), STEPS, batch, loss, replay, beat)
                updates += n
            ref = next(r['scores']['recording'] for r in online['by_site'][held] if r['seed'] == seed)
            cap = next(r for r in motion['by_site'][held] if r['seed'] == seed)
            score = evaluate(held, seed, key, sites[held], banks[held], menu, frozen, nets, cuts(donors), ref, cap)
            result[held].append(dict(seed=seed, pairs=sum(len(v) for v in source), donor_audit=checks, **score))
            beat(phase='cell', held=held, seed=seed, updates=updates)
    path = OUT / 'head_freeze.json'
    run.once(path, run.stable(dict(registration=reg, checkpoints=checkpoints)))
    summary = summarize(result)
    run.once(OUT / 'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(path),
        summary=summary, by_site=result, original_2pct=True, external_outcome_access=False, deployment_changed=False,
        final_fitted_updates=len(checkpoints) * STEPS, offline_matching=True, submission_ready=False)))
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
