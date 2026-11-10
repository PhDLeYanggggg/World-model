"""Fit and score a one-exchange ranker at the frozen OpenDD budget."""

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
from scripts import audit_m3w_opendd_exchange_frontier as frontier
from src.world_model import m3w_one_exchange_ranker as model


run, old, study, parent, np, torch = frontier.run, frontier.old, frontier.study, frontier.parent, frontier.np, frontier.torch
base, prefix = frontier.base, frontier.prefix
OUT = frontier.OUT.parent / 'opendd_one_exchange_ranker_v1'
PRIVATE = frontier.PRIVATE.parent / OUT.name
STEPS, BATCH, SAMPLE = 400, 128, 32
PRIMARY = 'context_guarded'
ARMS = ('frozen', 'min_movement', 'local_guarded', PRIMARY, 'context_no_gain_guard', 'context_forced')


def register():
    key, motion_key, cells, ranks, heads, baseline, motion = frontier.register()
    path = frontier.OUT / 'readout.json'
    bound = run.read(path)
    if (bound['registration'] != key or not bound['hindsight_only'] or bound['external_outcome_access']
            or not bound['original_2pct'] or bound['summary']['bounds']['one']['positive'] != [178, 183]):
        raise ValueError('Require the unchanged one-exchange diagnostic')
    inputs = [frontier.OUT / 'registration.json', path, OUT / 'protocol.md', Path(__file__), Path(model.__file__)]
    p = OUT / 'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(x) for x in inputs], steps=STEPS, batch=BATCH,
        sample_per_anchor=SAMPLE, threads=4, workers=0, forecast_steps=[8, 12], seeds=study.prior.SEEDS,
        primary=PRIMARY, score_margin=.10, original_2pct=True, no_held_tuning=True, external_outcome_access=False)))
    return run.sha(p), key, motion_key, cells, ranks, heads, baseline, motion


def indices(d, name, rows):
    ai, qi = d['rec'] == name, d['recordings'] == name
    here, r = np.flatnonzero(ai), rows[qi].copy()
    have = r >= 0
    if (not np.array_equal(np.unique(d['query'][ai]), np.flatnonzero(qi))
            or not np.isin(r[have], here).all() or (np.diff(d['starts'][qi]) <= 0).any()):
        raise ValueError('Changed recording order or nominee assignment')
    r[have] = np.searchsorted(here, r[have])
    return ai, qi, r


def measure(d, ai, bank, path, rows, count, cut):
    cv, ols = (z[ai] for z in bank[:2])
    p = old.prediction(cv, path[ai], rows, count)
    score = parent.score({k: d[k][ai] for k in ('y', 'valid', 'rec')}, cv, ols, p, cut)
    if score['edits'] != int(count.sum()):
        raise ValueError('Scored path edits do not match the plan')
    return p, score


def donor(file, identity, d, bank, menu, cut, seed):
    x, legal, rows, path, _ = menu
    cv = bank[0]
    move = base.model.movement_prefix(cv, path, rows)
    meta = file.with_suffix('.json')
    if not meta.exists():
        rng = np.random.default_rng(seed + 913)
        data, event, gain, stats = [], [], [], Counter()
        for name in sorted(set(d['rec'])):
            ai, qi, r = indices(d, name, rows)
            l, m, z = legal[qi], move[qi], x[qi]
            y, valid = d['y'][ai], d['valid'][ai]
            mass = np.linalg.norm(cv[ai] - y, axis=-1)[valid].sum()
            if mass <= 0.:
                continue
            schedules = (np.tile([0., 1., -1.], (len(l), 1)),
                         np.column_stack((np.zeros(len(l)), rng.standard_normal((len(l), 2)))))
            visited = set()
            for scores in schedules:
                a = study.model.streaming_count(scores, l, d['recordings'][qi], d['starts'][qi], study.RATE)
                if tuple(a) in visited:
                    continue
                visited.add(tuple(a))
                cap = float(m[np.arange(len(a)), a].sum())
                pairs = model.exchange_menu(a, l, m, cap)
                if not len(pairs):
                    continue
                stats.update(anchors=1, legal_pairs=len(pairs))
                if len(pairs) > SAMPLE:
                    pairs = pairs[np.sort(rng.choice(len(pairs), size=SAMPLE, replace=False))]
                f, _ = model.features(z, l, m, a, cap, pairs)
                anchor, old_score = measure(d, ai, bank, path, r, a, cut)
                data.append(f)
                for i, j in pairs:
                    c = model.exchanged(a, i, j)
                    proposed, new_score = measure(d, ai, bank, path, r, c, cut)
                    event.append(1 + new_score['dual_positive'] - old_score['dual_positive'])
                    gain.append(float(100. * parent.transfer.paired(anchor, proposed, y, valid)[1].sum() / mass))
        width = 4*x.shape[1] + 9
        a = np.concatenate(data) if data else np.empty((0, width), dtype=np.float32)
        b = np.asarray(event, dtype=int)
        c = np.clip(np.asarray(gain, dtype=np.float32), -1., 1.)
        if (a.shape != (len(b), width) or not np.isfinite(a).all() or not np.isfinite(c).all() or len(b) != len(c)):
            raise ValueError('Nonfinite donor training examples')
        tmp = file.with_suffix('.pending')
        with tmp.open('wb') as f:
            np.savez_compressed(f, x=a, event=b, gain=c)
        tmp.replace(file)
        stats.update(sampled=len(b), rescued=int((b == 2).sum()), destroyed=int((b == 0).sum()))
        run.once(meta, run.stable(dict(identity=identity, cache_sha256=run.sha(file), audit=dict(stats))))
    r = run.read(meta)
    if r['identity'] != identity or r['cache_sha256'] != run.sha(file):
        raise ValueError('Cached training data changed role or contents')
    with np.load(file, allow_pickle=False) as z:
        tensors = tuple(torch.from_numpy(z[k]) for k in ('x', 'event', 'gain'))
    if len(tensors[0]) != r['audit']['sampled']:
        raise ValueError('Donor count changed')
    return tensors, r['audit']


def batcher(data):
    sources = [v for v in data if len(v[0])]
    if not sources:
        raise ValueError('No feasible donor exchange examples')

    def batch(rng):
        n = np.bincount(rng.integers(len(sources), size=BATCH), minlength=len(sources))
        take = [rng.integers(len(x[0]), size=m) for x, m in zip(sources, n, strict=True)]
        return tuple(torch.cat([s[k][i] for s, i in zip(sources, take, strict=True)]) for k in range(3))

    return batch


def evaluate(site, seed, key, motion_key, d, bank, menu, frozen, nets, cut, original, diagnostic):
    cv, ols = bank[:2]
    x, legal, rows, path, _ = menu
    move = base.model.movement_prefix(cv, path, rows)
    logits = frozen(torch.from_numpy(x), torch.from_numpy(legal)).detach().numpy()
    online = study.model.streaming_count(logits, legal, d['recordings'], d['starts'], study.RATE)
    if parent.score(d, cv, ols, old.prediction(cv, path, rows, online), cut) != original:
        raise ValueError('Frozen streaming scorer changed')
    predictions = {k: cv.copy() for k in ARMS}
    transitions = {k: Counter({'exchanged': 0, 'rescued': 0, 'destroyed': 0}) for k in ARMS}
    coverage = Counter({'eligible': 0, 'candidates': 0, 'verified_hindsight_rescues': 0, 'hindsight_rescue_reached': 0})
    for i, name in enumerate(sorted(set(d['rec']))):
        ai, qi, r = indices(d, name, rows)
        m, l, z = move[qi], legal[qi], x[qi]
        tag = f'{site}_{seed}_{i:04d}.json'
        ident = dict(site=site, seed=seed, ordinal=i)
        cached = run.read(base.PRIVATE / tag)
        bound = run.read(frontier.PRIVATE / tag)
        if (cached['identity'] != dict(registration=motion_key, **ident)
                or bound['identity'] != dict(registration=key, **ident)):
            raise ValueError('Exchange audit uses an unbound frozen plan')
        a = np.asarray(cached['score_plan'])
        cap = prefix.saved_plan(m, l, online[qi], a)
        if cap != cached['cap'] or cap != bound['cap']:
            raise ValueError('Original movement cap did not reproduce')
        pairs = model.exchange_menu(a, l, m, cap)
        p0, s0 = measure(d, ai, bank, path, r, a, cut)
        if s0['dual_positive'] != bound['anchor_dual']:
            raise ValueError('Frontier anchor dual pass changed')
        feasible = bound['results']['one']
        proven = not s0['dual_positive'] and (feasible['lower_pp'] or 0.) > 1e-6
        coverage.update(eligible=int(bool(len(pairs))), candidates=len(pairs), verified_hindsight_rescues=int(proven))
        take = dict.fromkeys(ARMS)
        if len(pairs):
            f, delta = model.features(z, l, m, a, cap, pairs)
            with torch.inference_mode():
                context = nets['context'](torch.from_numpy(f))
                f[:, -(2*x.shape[1]+3):] = 0.
                local = nets['local'](torch.from_numpy(f))
            take['min_movement'] = int(delta.argmin()) if delta.min() < -1e-12 else None
            take['local_guarded'] = model.rank(local)
            take[PRIMARY] = model.rank(context)
            take['context_no_gain_guard'] = model.rank(context, guard=False)
            take['context_forced'] = model.rank(context, margin=-np.inf, guard=False)
        for k, j in take.items():
            if j is None:
                predictions[k][ai] = p0
                continue
            c = model.exchanged(a, *pairs[j])
            prefix.saved_plan(m, l, online[qi], c)
            p, score = measure(d, ai, bank, path, r, c, cut)
            if score['dual_positive'] and min(score['vs_cv'], score['vs_ols4']) > 1e-6 and (
                    feasible['infeasible'] or (feasible['upper_pp'] is not None and feasible['upper_pp'] <= 1e-6)):
                raise ValueError('One exchange contradicts a certified upper exclusion')
            gain = score['dual_positive'] - s0['dual_positive']
            transitions[k].update(exchanged=1, rescued=int(gain > 0), destroyed=int(gain < 0))
            if k == PRIMARY:
                coverage['hindsight_rescue_reached'] += int(proven and gain > 0)
            predictions[k][ai] = p
    summary = {k: parent.score(d, cv, ols, p, cut) for k, p in predictions.items()}
    if (any(v['edits'] != diagnostic['edits'] for v in summary.values())
            or (summary['frozen']['dual_positive'], summary['frozen']['risk_pass'], summary['frozen']['undefined'])
            != (diagnostic['score_plan'], diagnostic['score_risk'], diagnostic['score_undefined'])):
        raise ValueError('Matched edit counts or frozen score changed')
    if any(v['dual_positive'] != summary['frozen']['dual_positive'] + transitions[k]['rescued'] - transitions[k]['destroyed']
           for k, v in summary.items()):
        raise ValueError('One-exchange transitions did not reconcile')
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv - y, axis=-1)[valid].sum()
    effect = {k: float(100. * parent.transfer.paired(predictions['frozen'], p, y, valid)[1].sum()/mass)
              for k, p in predictions.items() if k != 'frozen'}
    return dict(scores=summary, transitions={k: dict(v) for k, v in transitions.items()}, coverage=dict(coverage),
                paired_lower_pp=effect)


def summarize(cells):
    flat = [r for group in cells.values() for r in group]
    draws = np.random.default_rng(817).integers(len(cells), size=(3000, len(cells)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in cells.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    locations_positive=int((x > 0.).sum()))

    out = {}
    for k in ARMS:
        v = {f: sum(r['scores'][k][f] for r in flat) for f in ('views', 'edits', 'dual_positive', 'risk_pass', 'undefined')}
        v.update({f: sum(r['transitions'][k][f] for r in flat) for f in ('exchanged', 'rescued', 'destroyed')})
        if k != 'frozen':
            v['gain_minus_frozen_pp'] = ci(lambda r: r['paired_lower_pp'][k])
            v['pass_rate_minus_frozen_pp'] = ci(lambda r: 100.*(r['scores'][k]['dual_positive']
                - r['scores']['frozen']['dual_positive'])/r['scores']['frozen']['views'])
        out[k] = v
    coverage = sum((Counter(r['coverage']) for r in flat), Counter())
    examples = [sum(v['sampled'] for v in r['donor_audit'].values()) for r in flat]
    return dict(arms=out, coverage=dict(coverage), donor_examples_across_cells=sum(examples),
                donor_examples_per_cell=[min(examples), max(examples)])


def execute(reg, key, motion_key, cells, ranks, heads, original, motion, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        f = parent.prior.prior.old.source.prior.PRIVATE / f'{s}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    result, checkpoints, updates = {}, {}, 0
    (PRIVATE / 'donors').mkdir(exist_ok=True)

    def cuts(roster):
        return np.quantile(np.concatenate([sites[s]['cut_samples'] for s in roster]), [.25, .75])

    rank_net = lambda: study.prior.model.SwapValue(False)
    for held in sites:
        result[held] = []
        donors = [s for s in sites if s != held]
        for row in cells['by_site'][held]:
            seed = row['seed']
            table, bindings, audit = [], {}, {}
            for s in donors:
                omitted = sorted((held, s))
                fit = [t for t in sites if t not in omitted]
                meta = heads['rank_heads']['rank_' + '_'.join(omitted) + f'_{seed}']
                if meta['identity'] != dict(registration=heads['registration'], omitted=omitted, fit=fit, seed=seed):
                    raise ValueError('Donor ranker used a forbidden location')
                rank = study.load(meta, rank_net, study.RANK_STEPS)
                menu = prefix.menu_for(sites[s], banks[s], rank)
                name = f'{"_".join(omitted)}_{seed}_{s}'
                path = PRIVATE / 'donors' / (name + '.npz')
                identity = dict(registration=reg, omitted=omitted, donor=s, seed=seed, rank=meta['artifact']['sha256'])
                batch, audit[s] = donor(path, identity, sites[s], banks[s], menu, cuts(fit), seed)
                table.append(batch)
                bindings[s] = run.sha(path.with_suffix('.json'))
                beat(phase='donor', held=held, seed=seed, donor=s, examples=audit[s]['sampled'])
            rank = study.load(ranks['heads'][f'{held}/{seed}/additive_rank'], rank_net, study.prior.STEPS)
            menu = prefix.menu_for(sites[held], banks[held], rank)
            w, nets = menu[0].shape[1], {}
            for arm in ('local', 'context'):
                name = f'{held}_{seed}_{arm}'
                identity = dict(registration=reg, held=held, train=donors, seed=seed, arm=arm, data=bindings)

                def loss(net, x, y, g, local=arm == 'local'):
                    if local:
                        x = x.clone()
                        x[:, -(2*w+3):] = 0.
                    return model.objective(net(x), y, g)

                nets[arm], checkpoints[name], n = study.fit(identity, str(PRIVATE / name), seed,
                    lambda: model.ExchangeRanker(4*w+9), STEPS, batcher(table), loss, replay, beat)
                updates += n
            frozen = study.load(heads['count_heads'][f'count_{held}_{seed}_recording'],
                                lambda: study.model.CountHead(w), study.HEAD_STEPS)
            ref = next(v['scores']['recording'] for v in original['by_site'][held] if v['seed'] == seed)
            cap = next(v for v in motion['by_site'][held] if v['seed'] == seed)
            metrics = evaluate(held, seed, key, motion_key, sites[held], banks[held], menu, frozen, nets, cuts(donors), ref, cap)
            result[held].append(dict(seed=seed, donor_audit=audit, **metrics))
            beat(phase='cell', held=held, seed=seed, updates=updates)
    p = OUT / 'head_freeze.json'
    run.once(p, run.stable(dict(registration=reg, checkpoints=checkpoints)))
    summary = summarize(result)
    if any((a['views'], a['edits']) != (606, 780) for a in summary['arms'].values()):
        raise ValueError('Unexpected matched evaluation cohort')
    run.once(OUT / 'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(p), summary=summary,
        by_site=result, offline_matching=True, deployment_changed=False, external_outcome_access=False, original_2pct=True,
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
