"""Test aligned observed map and peer context at exactly the frozen spend."""

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
from scripts import run_m3w_opendd_map_direction as road
from scripts import run_m3w_opendd_one_exchange_ranker as before
from src.world_model import m3w_exchange_context as context


run, model, base, parent, prefix, old, np, torch = (before.run, before.model, before.base, before.parent, before.prefix,
    before.old, before.np, before.torch)
study = before.study
OUT = before.OUT.parent/'opendd_exchange_context_v1'
PRIVATE = before.PRIVATE.parent/OUT.name
ARMS = ('frozen', *context.MODES)
CONTRASTS = (('aligned', 'motion'), ('aligned', 'misaligned'), ('map', 'motion'), ('peer', 'motion'), ('aligned', 'frozen'))


def register():
    old_key, key, motion_key, cells, ranks, heads, original, motion = before.register()
    p = before.OUT/'readout.json'
    past = run.read(p)
    assets = run.read(road.OUT/'registration.json')['static_assets']
    if past['registration'] != old_key or past['external_outcome_access'] or not past['original_2pct']:
        raise ValueError('Require the earlier verified exchange study')
    f = OUT/'registration.json'
    files = [p, before.OUT/'registration.json', road.OUT/'registration.json', OUT/'protocol.md',
             Path(__file__), Path(context.__file__)]
    run.once(f, run.stable(dict(inputs=[run.artifact(q) for q in files], assets=assets, arms=context.MODES,
        steps=before.STEPS, seeds=study.prior.SEEDS, workers=0, threads=4, original_2pct=True,
        external_outcome_access=False, no_held_tuning=True)))
    return run.sha(f), old_key, key, motion_key, cells, ranks, heads, original, motion, past, assets


def design(x, width, mode):
    if mode == 'misaligned':
        x = x.clone()
        w = width+2*context.WIDTH
        for start in (0, w, 2*w+6, 3*w+6):
            i = start+width
            x[:, i:i+context.WIDTH] = x[:, i+context.WIDTH:i+2*context.WIDTH]
    return x*torch.from_numpy(context.mask(width, mode))


def augment(d, bank, menu, raster):
    x, legal, rows, path, scale = menu
    a, b = context.descriptors(d['h'], bank[0], path, d['query'], rows, raster)
    c, w, support = context.aligned_and_rotated(a, b, d['recordings'])
    return (np.column_stack((x, c, w)), legal, rows, path, scale), support


def evaluate(s, seed, key, motion_key, d, bank, menu, support, width, frozen, nets, cut, original, cap):
    cv, ols = bank[:2]
    x, legal, rows, path, _ = menu
    with torch.inference_mode():
        logits = frozen(torch.from_numpy(x[:, :width]), torch.from_numpy(legal)).numpy()
    online = study.model.streaming_count(logits, legal, d['recordings'], d['starts'], study.RATE)
    if parent.score(d, cv, ols, old.prediction(cv, path, rows, online), cut) != original:
        raise ValueError('Archived streaming baseline changed')
    movement = base.model.movement_prefix(cv, path, rows)
    pred = {k: cv.copy() for k in ARMS}
    transitions = {k: Counter({'exchanged': 0, 'rescued': 0, 'destroyed': 0}) for k in ARMS}
    coverage = Counter({'candidates': 0, 'either_query_supported': 0, 'both_queries_supported': 0})
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = before.indices(d, rec, rows)
        tag = f'{s}_{seed}_{i:04d}.json'
        m, l, z = movement[q], legal[q], x[q]
        v = run.read(base.PRIVATE/tag)
        if v['identity'] != dict(registration=motion_key, site=s, seed=seed, ordinal=i):
            raise ValueError('Unbound frozen allocation')
        anchor = np.asarray(v['score_plan'])
        budget = prefix.saved_plan(m, l, online[q], anchor)
        if budget != v['cap']:
            raise ValueError('Frozen movement cap changed')
        paths, stats = before.measure(d, ai, bank, path, r, anchor, cut)
        pairs = model.exchange_menu(anchor, l, m, budget)
        frontier = run.read(before.frontier.PRIVATE/tag)
        if frontier['identity'] != dict(registration=key, site=s, seed=seed, ordinal=i):
            raise ValueError('Unbound exchange certificate')
        bound = frontier['results']['one']
        choices = dict.fromkeys(ARMS)
        if len(pairs):
            t = support[q][pairs]
            coverage.update(candidates=len(pairs), either_query_supported=int(t.any(1).sum()),
                            both_queries_supported=int(t.all(1).sum()))
            f = torch.from_numpy(model.features(z, l, m, anchor, budget, pairs)[0])
            with torch.inference_mode():
                for arm, net in nets.items():
                    choices[arm] = model.rank(net(design(f, width, arm)))
        for arm, i0 in choices.items():
            if i0 is None:
                pred[arm][ai] = paths
                continue
            c = model.exchanged(anchor, *pairs[i0])
            prefix.saved_plan(m, l, online[q], c)
            p, score = before.measure(d, ai, bank, path, r, c, cut)
            if score['dual_positive'] and min(score['vs_cv'], score['vs_ols4']) > 1e-6 and (
                    bound['infeasible'] or (bound['upper_pp'] is not None and bound['upper_pp'] <= 1e-6)):
                raise ValueError('Learned action disagrees with the exchange certificate')
            delta = score['dual_positive'] - stats['dual_positive']
            transitions[arm].update(exchanged=1, rescued=int(delta > 0), destroyed=int(delta < 0))
            pred[arm][ai] = p
    scores = {k: parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    if (any(v['edits'] != cap['edits'] for v in scores.values())
            or (scores['frozen']['dual_positive'], scores['frozen']['risk_pass']) != (cap['score_plan'], cap['score_risk'])
            or any(v['dual_positive'] != scores['frozen']['dual_positive'] + transitions[k]['rescued'] - transitions[k]['destroyed']
                   for k, v in scores.items())):
        raise ValueError('Matched outcomes did not reconcile')
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effect = {f'{a}_minus_{b}': dict(dual_pp=100.*(scores[a]['dual_positive'] - scores[b]['dual_positive'])/scores[b]['views'],
        gain_pp=float(100.*parent.transfer.paired(pred[b], pred[a], y, valid)[1].sum()/mass)) for a, b in CONTRASTS}
    return dict(scores=scores, effects=effect, coverage=dict(coverage), transitions={k: dict(v) for k, v in transitions.items()})


def summarize(groups):
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    flat = [v for g in groups.values() for v in g]

    def ci(k, f):
        x = np.array([np.mean([r['effects'][k][f] for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    locations_positive=int((x > 0.).sum()))

    summary = {}
    for k in ARMS:
        arm = {f: sum(r['scores'][k][f] for r in flat) for f in ('views', 'edits', 'dual_positive', 'risk_pass', 'undefined')}
        arm.update({f: sum(r['transitions'][k][f] for r in flat) for f in ('exchanged', 'rescued', 'destroyed')})
        summary[k] = arm
    return dict(arms=summary, comparisons={k: {f: ci(k, f) for f in ('dual_pp', 'gain_pp')} for k in flat[0]['effects']},
                support=dict(sum((Counter(r['coverage']) for r in flat), Counter())))


def execute(reg, previous, key, motion_key, cells, ranks, heads, original, motion, past, assets, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        f = parent.prior.prior.old.source.prior.PRIVATE/f'{s}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    maps = {s: road.raster_for(s, assets[s])[0] for s in sites}
    result, checkpoints, updates = {}, {}, 0
    (PRIVATE/'donors').mkdir(exist_ok=True)

    def cuts(roster):
        return np.quantile(np.concatenate([sites[s]['cut_samples'] for s in roster]), [.25, .75])

    rank_net = lambda: study.prior.model.SwapValue(False)
    for held in sites:
        result[held] = []
        donors = [s for s in sites if s != held]
        for row in cells['by_site'][held]:
            seed = row['seed']
            ref_past = next(v for v in past['by_site'][held] if v['seed'] == seed)
            table, bindings = [], {}
            for s in donors:
                omitted = sorted((held, s))
                fit = [t for t in sites if t not in omitted]
                meta = heads['rank_heads']['rank_' + '_'.join(omitted) + f'_{seed}']
                if meta['identity'] != dict(registration=heads['registration'], omitted=omitted, fit=fit, seed=seed):
                    raise ValueError('Donor ranker uses a held site')
                rank = study.load(meta, rank_net, study.RANK_STEPS)
                menu, _ = augment(sites[s], banks[s], prefix.menu_for(sites[s], banks[s], rank), maps[s])
                name = f'{"_".join(omitted)}_{seed}_{s}'
                path = PRIVATE/'donors'/(name+'.npz')
                identity = dict(registration=reg, omitted=omitted, donor=s, seed=seed, rank=meta['artifact']['sha256'])
                batch, audit = before.donor(path, identity, sites[s], banks[s], menu, cuts(fit), seed)
                old_data = before.PRIVATE/'donors'/(name+'.npz')
                old_receipt = run.read(old_data.with_suffix('.json'))
                if old_receipt['identity'] != dict(identity, registration=previous) or old_receipt['cache_sha256'] != run.sha(old_data):
                    raise ValueError('Unbound baseline training samples')
                with np.load(old_data, allow_pickle=False) as z:
                    if audit != ref_past['donor_audit'][s] or any(not np.array_equal(z[k], a.numpy())
                            for k, a in zip(('event', 'gain'), batch[1:], strict=True)):
                        raise ValueError('Context input changed the donor labels or sampled actions')
                table.append(batch)
                bindings[s] = run.sha(path.with_suffix('.json'))
                beat(phase='donor', held=held, seed=seed, donor=s)
            rank = study.load(ranks['heads'][f'{held}/{seed}/additive_rank'], rank_net, study.prior.STEPS)
            m = prefix.menu_for(sites[held], banks[held], rank)
            width = m[0].shape[1]
            menu, support = augment(sites[held], banks[held], m, maps[held])
            batch, nets = before.batcher(table), {}
            for k in context.MODES:
                name = f'{held}_{seed}_{k}'
                ident = dict(registration=reg, held=held, train=donors, seed=seed, arm=k, data=bindings)

                def loss(net, x, event, gain, k=k):
                    return model.objective(net(design(x, width, k)), event, gain)

                nets[k], checkpoints[name], n = study.fit(ident, str(PRIVATE/name), seed,
                    lambda: model.ExchangeRanker(4*menu[0].shape[1]+9), before.STEPS, batch, loss, replay, beat)
                updates += n
            frozen = study.load(heads['count_heads'][f'count_{held}_{seed}_recording'],
                                lambda: study.model.CountHead(width), study.HEAD_STEPS)
            ref = next(v['scores']['recording'] for v in original['by_site'][held] if v['seed'] == seed)
            cap = next(v for v in motion['by_site'][held] if v['seed'] == seed)
            metrics = evaluate(held, seed, key, motion_key, sites[held], banks[held], menu, support, width, frozen,
                               nets, cuts(donors), ref, cap)
            result[held].append(dict(seed=seed, **metrics))
            beat(phase='cell', held=held, seed=seed, updates=updates)
    p = OUT/'head_freeze.json'
    run.once(p, run.stable(dict(registration=reg, checkpoints=checkpoints)))
    summary = summarize(result)
    if summary['support']['candidates'] != 7198 or any((a['views'], a['edits']) != (606, 780) for a in summary['arms'].values()):
        raise ValueError('Context experiment changed the frozen exchange menu')
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(p), summary=summary,
        by_site=result, offline_matching=True, deployment_changed=False, external_outcome_access=False,
        original_2pct=True, final_fitted_updates=before.STEPS*len(checkpoints), submission_ready=False)))
    return updates, summary


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

        updates, summary = execute(*register(), args.replay, beat)
        beat(phase='complete', updates=updates, replay=args.replay, summary=summary)


if __name__ == '__main__':
    main()
