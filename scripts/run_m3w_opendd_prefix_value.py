"""Fit a continuous prefix-value head without changing forecasts or the scorer."""

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
from scripts import audit_m3w_opendd_count_motion as prior
from src.world_model import m3w_prefix_value as model


old, run, study, parent, np, torch = prior.old, prior.run, prior.study, prior.parent, prior.np, prior.torch
OUT = prior.OUT.parent / 'opendd_prefix_value_v1'
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS, BATCH = 500, 128
NEW = ('gain_only', 'joint_no_penalty', 'joint_value')
ARMS = ('frozen', *NEW)
PRIMARY = 'joint_value'


def register():
    key, _, cells, ranks, heads, baseline, _ = prior.register()
    prev = prior.OUT / 'readout.json'
    budget = run.read(prev)
    if (budget['registration'] != key or budget['confirmation_access'] or not budget['original_2pct']
            or budget['summary']['score_unresolved']):
        raise ValueError('Require the verified realized movement comparison')
    files = [prior.OUT / 'registration.json', prev, OUT / 'protocol.md', Path(__file__), Path(model.__file__)]
    path = OUT / 'registration.json'
    run.once(path, run.stable(dict(inputs=[run.artifact(f) for f in files], forecast_steps=[8, 12],
        seeds=study.prior.SEEDS, updates=STEPS, batch=BATCH, threads=4, workers=0, original_2pct=True,
        confirmation_access=False, no_held_tuning=True, primary=PRIMARY, matched_budget='frozen_count_and_movement_cap')))
    return run.sha(path), key, cells, ranks, heads, baseline, budget


def menu_for(d, bank, rank):
    menu = study.menus(d, bank, rank)
    legal, row, arm = (menu[k] for k in ('legal', 'row', 'arm'))
    path = bank[0].copy()
    use = row >= 0
    path[row[use]] = bank[2][row[use], arm[use]]
    # The bridge feature scale uses observed motion only.
    scale = np.array([12. * np.median(bank[4][ix]) for ix, _ in study.prior.groups(d)])
    return menu['x'], legal, row, path, scale


def training(d, bank, menu, cut):
    x, legal, row, path, scale = menu
    g, r, _, _, _, _, _ = old.prefix_terms(bank[0], bank[1], path, d['y'], d['valid'], row, legal, cut)
    y = (np.concatenate((g, r), axis=-1)[:, 1:] / scale[:, None, None]).astype(np.float32)
    use = legal[:, 1]
    return tuple(torch.from_numpy(a[use]) for a in (x, y, legal[:, 1:]))


def batcher(data):
    sources = [s for s in data if len(s[0])]
    if not sources:
        raise ValueError('No eligible donor actions')

    def batch(rng):
        n = np.bincount(rng.integers(len(sources), size=BATCH), minlength=len(sources))
        ids = [rng.integers(len(x[0]), size=k) for k, x in zip(n, sources, strict=True)]
        return tuple(torch.cat([a[k][i] for a, i in zip(sources, ids, strict=True)]) for k in range(3))

    return batch


def saved_plan(movement, legal, online, proposal):
    q = np.arange(len(online))
    cap = float(movement[q, online].sum())
    if (proposal.shape != online.shape or not np.issubdtype(proposal.dtype, np.integer)
            or (proposal < 0).any() or (proposal > 2).any() or proposal.sum() != online.sum()
            or not legal[q, proposal].all() or not prior.model.respects_tokens(proposal, legal[:, 1])
            or not prior.model.within_cap(movement[q, proposal].sum(), cap)):
        raise ValueError('Matched proposal does not preserve realized budget')
    return cap


def matched(site, seed, key, d, cv, path, menu, online, scores):
    _, legal, rows, _, _ = menu
    out = {name: online.copy() for name in scores}
    records = d['recordings']
    moves = prior.model.movement_prefix(cv, path, rows)
    statuses = Counter({k: 0 for k in scores})
    for i, name in enumerate(sorted(set(d['rec']))):
        ix = records == name
        legal0, m, c = legal[ix], moves[ix], online[ix]
        if not np.array_equal(np.unique(d['query'][d['rec'] == name]), np.flatnonzero(ix)):
            raise ValueError('Matched recording roster changed')
        row = run.read(prior.PRIVATE / f'{site}_{seed}_{i:04d}.json')
        if row['identity'] != dict(registration=key, site=site, seed=seed, ordinal=i) or row['edits'] != int(c.sum()):
            raise ValueError('Frozen score plan is not bound to the original count and site')
        p = np.asarray(row['score_plan'])
        cap = saved_plan(m, legal0, c, p)
        if cap != row['cap']:
            raise ValueError('Frozen movement cap changed')
        out['frozen'][ix] = p
        for label in NEW:
            p, status = prior.model.allocate(legal0, m, int(c.sum()), cap, score=scores[label][ix])
            if p is None:
                # Preserve the baseline and disclose the unresolved match.
                statuses[label] += 1
                continue
            saved_plan(m, legal0, c, p)
            j = np.arange(len(c))
            if scores[label][ix][j, p].sum() + 1e-8 < scores[label][ix][j, c].sum():
                raise ValueError(f'{label} optimizer lost to the online feasible plan')
            out[label][ix] = p
    return out, dict(statuses)


def evaluate(site, seed, key, d, bank, menu, frozen, nets, cut, original, diagnostic):
    cv, ols = bank[:2]
    x, legal, rows, path, scale = menu
    with torch.inference_mode():
        xx = torch.from_numpy(x)
        scores = {'frozen': frozen(xx, torch.from_numpy(legal)).numpy()}
        y = nets['gain_only'](xx).numpy()
        z = nets['joint'](xx).numpy()
        scores['gain_only'] = model.decision_scores(y, scale, legal, penalty=False)
        scores['joint_no_penalty'] = model.decision_scores(z, scale, legal, penalty=False)
        scores['joint_value'] = model.decision_scores(z, scale, legal, penalty=True)
    counts = {k: study.model.streaming_count(s, legal, d['recordings'], d['starts'], study.RATE) for k, s in scores.items()}
    allocations, unresolved = matched(site, seed, key, d, cv, path, menu, counts['frozen'], scores)
    mode = dict(online=counts, matched=allocations)
    target, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv - target, axis=-1)[valid].sum()
    result = {}
    for setting, arms in mode.items():
        pred = {k: old.prediction(cv, path, rows, c) for k, c in arms.items()}
        summary = {k: parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
        if setting == 'online' and summary['frozen'] != original:
            raise ValueError('Archived online comparator did not reproduce')
        if setting == 'matched':
            if any(summary[k]['edits'] != diagnostic['edits'] for k in ARMS):
                raise ValueError('Matched arms changed the original edit count')
            for a, b in (('dual_positive', 'score_plan'), ('risk_pass', 'score_risk'), ('undefined', 'score_undefined')):
                if summary['frozen'][a] != diagnostic[b]:
                    raise ValueError('Archived label-blind comparator did not reproduce')
        direct = {k: float(100. * parent.transfer.paired(pred['frozen'], pred[k], target, valid)[1].sum() / mass)
                  for k in NEW}
        result[setting] = dict(scores=summary, paired_lower_pp=direct)
    return dict(**result, matched_unresolved=unresolved)


def summarize(result):
    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(result), size=(3000, len(result)))

    def ci(fn):
        means = np.array([np.mean([fn(r) for r in rows]) for rows in result.values()])
        return dict(point=float(means.mean()), interval95=np.quantile(means[draw].mean(1), [.025, .975]).tolist(),
                    positive_locations=int((means > 0.).sum()))

    summary = {}
    for mode in ('online', 'matched'):
        fields = ('views', 'edits', 'dual_positive', 'risk_pass', 'undefined', 'unknown_edits')
        summary[mode] = {k: {f: sum(r[mode]['scores'][k][f] for r in flat) for f in fields} for k in ARMS}
        for k in NEW:
            def dual_pp(r):
                a, b = r[mode]['scores'][k], r[mode]['scores']['frozen']
                return 100. * (a['dual_positive'] - b['dual_positive']) / b['views']
            summary[mode][k]['paired_pass_rate_pp'] = ci(dual_pp)
            summary[mode][k]['paired_conservative_gain_pp'] = ci(lambda r: r[mode]['paired_lower_pp'][k])
    summary['matched_unresolved'] = {k: sum(r['matched_unresolved'][k] for r in flat) for k in ARMS}
    return summary


def execute(reg, old_key, cells, ranks, heads, baseline, diagnostic, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        f = parent.prior.prior.old.source.prior.PRIVATE / f'{s}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    result, checkpoint = {}, {}
    updates = 0
    qcut = lambda roster: np.quantile(np.concatenate([sites[t]['cut_samples'] for t in roster]), [.25, .75])
    rank_net = lambda: study.prior.model.SwapValue(False)
    for held in sites:
        donors = [s for s in sites if s != held]
        outer_cut = qcut(donors)
        result[held] = []
        for cell in cells['by_site'][held]:
            seed = cell['seed']
            table, donor_hashes = [], {}
            for donor in donors:
                omit = sorted((held, donor))
                fit = [s for s in sites if s not in omit]
                name = 'rank_' + '_'.join(omit) + f'_{seed}'
                meta = heads['rank_heads'][name]
                if meta['identity'] != dict(registration=heads['registration'], omitted=omit, fit=fit, seed=seed):
                    raise ValueError('A donor ranker used the held or donor site')
                rank = study.load(meta, rank_net, study.RANK_STEPS)
                menu = menu_for(sites[donor], banks[donor], rank)
                table.append(training(sites[donor], banks[donor], menu, qcut(fit)[0]))
                donor_hashes[donor] = meta['artifact']['sha256']
            batch = batcher(table)
            width = table[0][0].shape[-1]
            nets = {}
            for arm in ('gain_only', 'joint'):
                name = f'{held}_{seed}_{arm}'
                ident = dict(registration=reg, held=held, seed=seed, train=donors, arm=arm, inner_hashes=donor_hashes)

                def loss(net, x, y, m, use_risk=arm == 'joint'):
                    return model.objective(net(x), y, m, risk=use_risk)

                nets[arm], checkpoint[name], n = study.fit(ident, str(PRIVATE / name), seed, lambda: model.PrefixValue(width),
                    STEPS, batch, loss, replay, beat)
                updates += n
            rk = study.load(ranks['heads'][f'{held}/{seed}/additive_rank'], rank_net, study.prior.STEPS)
            menu = menu_for(sites[held], banks[held], rk)
            frozen = study.load(heads['count_heads'][f'count_{held}_{seed}_recording'],
                                lambda: study.model.CountHead(width), study.HEAD_STEPS)
            ref = next(r['scores']['recording'] for r in baseline['by_site'][held] if r['seed'] == seed)
            cap = next(r for r in diagnostic['by_site'][held] if r['seed'] == seed)
            record = evaluate(held, seed, old_key, sites[held], banks[held], menu, frozen, nets, outer_cut, ref, cap)
            result[held].append(dict(seed=seed, **record))
            beat(phase='cell', held=held, seed=seed, updates=updates)
    path = OUT / 'head_freeze.json'
    run.once(path, run.stable(dict(registration=reg, checkpoints=checkpoint)))
    summary = summarize(result)
    run.once(OUT / 'readout.json', run.stable(dict(registration=reg, checkpoint_manifest=run.sha(path),
        summary=summary, by_site=result, original_2pct=True, confirmation_access=False, deployment_changed=False,
        final_fitted_updates=STEPS * len(checkpoint), submission_ready=False)))
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
            status = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **event)
            run.kin.prior.source.atomic(PRIVATE / 'heartbeat.json', status)
            if event['phase'] != 'fit':
                print(status, flush=True)

        updates, summary = execute(*register(), args.replay, beat)
        beat(phase='complete', updates=updates, replay=args.replay, summary=summary)


if __name__ == '__main__':
    main()
