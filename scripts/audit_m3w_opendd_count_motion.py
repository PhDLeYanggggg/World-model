"""Test whether fixed-count OpenDD opportunities survive a fixed movement cap."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_realized_stream_capacity as old
from src.evaluation import m3w_count_motion_capacity as model


run, study, parent, np, torch = old.run, old.study, old.parent, old.np, old.torch
OUT = old.OUT.parent / 'opendd_count_motion_v1'
PRIVATE = old.PRIVATE.parent / OUT.name
PASS = old.PASS


def register():
    previous, _, cells, ranks, heads, online, _ = old.register()
    readout = old.OUT / 'readout.json'
    frozen = run.read(readout)
    if frozen['registration'] != previous or frozen['confirmation_access'] or not frozen['original_2pct']:
        raise ValueError('Require the unchanged exact-count diagnostic')
    files = [old.OUT / 'registration.json', readout, OUT / 'protocol.md', Path(__file__), Path(model.__file__)]
    p = OUT / 'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], no_training=True,
        original_2pct=True, confirmation_access=False, steps=[8, 12], bootstrap_draws=3000,
        budget='exact_count_and_at_most_online_sum_of_agent_peak_displacement', bootstrap_seed=817)))
    return run.sha(p), previous, cells, ranks, heads, online, frozen


def evaluate(file, ident, archived, local, rows, logits, legal, online, name, cuts):
    cv, ols, chosen, y, valid = local
    query = np.arange(len(online))
    movement = model.movement_prefix(cv, chosen, rows)
    total, cap = int(online.sum()), float(movement[query, online].sum())

    def check(count):
        count = np.asarray(count)
        if (count.shape != online.shape or not np.issubdtype(count.dtype, np.integer)
                or (count < 0).any() or (count > 2).any() or count.sum() != total
                or not legal[query, count].all() or not model.respects_tokens(count, legal[:, 1])
                or not model.within_cap(movement[query, count].sum(), cap)):
            raise ValueError('Invalid recorded movement allocation')
        p = old.prediction(cv, chosen, rows, count)
        metrics = parent.score(dict(y=y, valid=valid, rec=np.full(len(cv), name)), cv, ols, p, cuts)
        if metrics['edits'] != total:
            raise ValueError('Actual edit count changed on original scoring')
        return metrics

    actual = check(online)
    was = archived['result']['status']
    if archived['total'] != total or archived['actual'] != bool(actual['dual_positive']):
        raise ValueError('Archived per-recording online score changed')

    if not file.exists():
        # Select using only the frozen scores and budget before inspecting utility.
        score_plan, score_status = model.allocate(legal, movement, total, cap, score=logits)
        g, r, e, _, off, easy_cap, mass = old.prefix_terms(*local, rows, legal, cuts[0])
        plan, status = model.allocate(legal, movement, total, cap, terms=(g, r, e, off, easy_cap, mass))
        if plan is not None:
            if check(plan)['dual_positive']:
                status = PASS
            else:
                plan, status = None, 'unresolved_numerical_check'
        if plan is None and actual['dual_positive']:
            if not status.startswith('unresolved'):
                raise ValueError('The original plan proves feasibility')
            plan, status = online, PASS
        run.once(file, run.stable(dict(identity=ident, edits=total, cap=cap, status=status,
                                      score_status=score_status, plan=None if plan is None else plan.tolist(),
                                      score_plan=None if score_plan is None else score_plan.tolist())))
    saved = run.read(file)
    if (saved['identity'], saved['edits'], saved['cap']) != (ident, total, cap):
        raise ValueError('Private receipt does not bind original spend')
    plan, score_plan, status = saved['plan'], saved['score_plan'], saved['status']
    if (status == PASS) != (plan is not None) or (plan is not None and not check(plan)['dual_positive']):
        raise ValueError('Original scorer did not verify the cached hindsight result')
    if (status == PASS and was != PASS) or (actual['dual_positive'] and status != PASS):
        raise ValueError('The new budget comparison is not nested')
    stats = check(score_plan) if score_plan is not None else None
    if score_plan is not None and (logits[query, score_plan].sum(dtype=float) + 1e-8
                                   < logits[query, online].sum(dtype=float)):
        raise ValueError('Score optimizer lost to the known feasible online plan')
    if stats is not None and stats['dual_positive'] and status not in (PASS, 'unresolved', 'unresolved_numerical_check'):
        raise ValueError('Score plan contradicts the hindsight feasibility claim')
    return Counter(views=1, edits=total, online=actual['dual_positive'], online_risk=actual['risk_pass'],
        online_undefined=actual['undefined'], count_only=was == PASS, count_motion=status == PASS,
        score_plan=0 if stats is None else stats['dual_positive'], score_risk=0 if stats is None else stats['risk_pass'],
        score_undefined=0 if stats is None else stats['undefined'], score_unresolved=stats is None,
        score_recovers=stats is not None and stats['dual_positive'] and not actual['dual_positive'],
        score_loses=stats is not None and actual['dual_positive'] and not stats['dual_positive'],
        score_changed=score_plan is not None and not np.array_equal(score_plan, online),
        **{f'hindsight_{status}': 1})


def effects(by_site):
    result = {}
    draw = np.random.default_rng(817).integers(len(by_site), size=(3000, len(by_site)))
    for a, b in (('count_motion', 'online'), ('score_plan', 'online'), ('count_only', 'count_motion')):
        d = np.array([np.mean([100. * (r[a] - r[b]) / r['views'] for r in v]) for v in by_site.values()])
        loo = (d.sum() - d) / (len(d) - 1)
        result[f'{a}_minus_{b}'] = dict(point_pp=float(d.mean()), ci95_pp=np.quantile(d[draw].mean(1), [.025, .975]).tolist(),
            leave_one_location_out_pp=[float(loo.min()), float(loo.max())], positive_locations=int((d > 0.).sum()))
    return result


def execute(reg, previous, cells, ranks, heads, frozen_online, frozen_count, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    results = {}
    for site, d in sites.items():
        src = parent.prior.prior.old.source.prior.PRIVATE / f'{site}_cohort.npz'
        with np.load(src, allow_pickle=False) as z:
            recordings, starts = z['recording'], z['start']
        bank = parent.prior.analytic(d)
        cv, ols, paths = bank[:3]
        q, rec, y, valid = (d[k] for k in ('query', 'rec', 'y', 'valid'))
        cuts = np.quantile(np.concatenate([v['cut_samples'] for s, v in sites.items() if s != site]), [.25, .75])
        results[site] = []
        for cell in cells['by_site'][site]:
            seed = cell['seed']
            rank = study.load(ranks['heads'][f'{site}/{seed}/additive_rank'],
                              lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            menu = study.menus(d, bank, rank)
            legal, rows, arms = (menu[k] for k in ('legal', 'row', 'arm'))
            net = study.load(heads['count_heads'][f'count_{site}_{seed}_recording'],
                             lambda: study.model.CountHead(menu['x'].shape[-1]), study.HEAD_STEPS)
            with torch.inference_mode():
                logits = net(torch.from_numpy(menu['x']), torch.from_numpy(legal)).numpy()
            online = study.model.streaming_count(logits, legal, recordings, starts, study.RATE)
            chosen = cv.copy()
            use = rows >= 0
            chosen[rows[use]] = paths[rows[use], arms[use]]
            prior_online = next(r['scores']['recording'] for r in frozen_online['by_site'][site] if r['seed'] == seed)
            if run.stable(parent.score(d, cv, ols, old.prediction(cv, chosen, rows, online), cuts)) != prior_online:
                raise ValueError('Frozen controller does not reproduce the archived site score')
            counts = Counter()
            for i, name in enumerate(sorted(set(rec))):
                ai, qi = rec == name, recordings == name
                here = np.flatnonzero(ai)
                r = rows[qi].copy()
                ok = r >= 0
                if (not np.array_equal(np.unique(q[ai]), np.flatnonzero(qi)) or (np.diff(starts[qi]) <= 0).any()
                        or not np.isin(r[ok], here).all()):
                    raise ValueError('Recording or query alignment changed')
                r[ok] = np.searchsorted(here, r[ok])
                tag = f'{site}_{seed}_{i:04d}.json'
                identity = dict(site=site, ordinal=i, seed=seed)
                archived = run.read(old.PRIVATE / tag)
                if archived['identity'] != dict(registration=previous, **identity):
                    raise ValueError('Unmatched count-only receipt')
                counts.update(evaluate(PRIVATE / tag, dict(registration=reg, **identity), archived,
                    (cv[ai], ols[ai], chosen[ai], y[ai], valid[ai]), r, logits[qi], legal[qi], online[qi], name, cuts))
                beat(site=site, seed=seed, completed=i + 1)
            ref = next(z for z in frozen_count['by_site'][site] if z['seed'] == seed)
            if [counts[k] for k in ('views', 'edits', 'online', 'count_only')] != [ref[k] for k in ('views', 'edits', 'online_pass', 'exact_pass')]:
                raise ValueError('Original comparison did not reproduce')
            results[site].append(dict(counts, seed=seed))
    fields = set.union(*(set(r) for v in results.values() for r in v)) - {'seed'}
    total = {k: sum(r.get(k, 0) for v in results.values() for r in v) for k in sorted(fields)}
    return dict(registration=reg, summary=total, by_site=results, contrasts=effects(results),
                original_2pct=True, confirmation_access=False, no_training=True, submission_ready=False)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**data):
            run.kin.prior.source.atomic(PRIVATE / 'heartbeat.json', dict(pid=os.getpid(),
                seconds=round(time.monotonic() - started, 3), **data))

        result = execute(*register(), beat)
        run.once(OUT / 'readout.json', run.stable(result))
        beat(phase='complete', updates=0, summary=result['summary'])
        print(result['summary'], result['contrasts'], flush=True)


if __name__ == '__main__':
    main()
