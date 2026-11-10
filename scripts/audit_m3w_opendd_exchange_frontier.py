"""Measure how much certified development gain is near the frozen schedule."""

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
from scripts import audit_m3w_opendd_count_motion as base
from scripts import run_m3w_opendd_prefix_value as prefix
from src.evaluation import m3w_exchange_frontier as solver


run, old, study, parent, np, torch = base.run, base.old, base.study, base.parent, base.np, base.torch
OUT = base.OUT.parent / 'opendd_exchange_frontier_v1'
PRIVATE = base.PRIVATE.parent / OUT.name
LEVELS = ('zero', 'one', 'two', 'unrestricted')
FLOORS = {'positive': 1e-6, '0.10pp': .1, '0.50pp': .5, '1.00pp': 1.}


def register():
    key, _, cells, ranks, heads, online, _ = base.register()
    p = base.OUT / 'readout.json'
    readout = run.read(p)
    if (readout['registration'] != key or readout['confirmation_access'] or not readout['original_2pct']
            or (readout['summary']['views'], readout['summary']['edits'], readout['summary']['score_plan'],
                readout['summary']['count_motion']) != (606, 780, 60, 210)):
        raise ValueError('Expected the unchanged matched-movement results')
    p0 = OUT / 'registration.json'
    files = [base.OUT / 'registration.json', p, OUT / 'protocol.md', Path(__file__), Path(solver.__file__)]
    run.once(p0, run.stable(dict(inputs=[run.artifact(f) for f in files], descriptive_gain_floors_pp=FLOORS,
        neighborhoods=LEVELS, seeds=study.prior.SEEDS, seconds_per_milp=2., original_2pct=True,
        external_outcome_access=False, no_training=True)))
    return run.sha(p0), key, cells, ranks, heads, online, readout


def intervals(rows, floors=FLOORS):
    """Count verified witnesses and views not yet excluded."""
    out = {}
    for key, f in floors.items():
        lower, upper = 0, 0
        for row in rows:
            lower += int(row['lower_pp'] is not None and row['lower_pp'] > f)
            upper += int(not row['infeasible'] and (row['upper_pp'] is None or row['upper_pp'] > f))
        if lower > upper:
            raise ValueError('Certified range contradicts a witness')
        out[key] = [lower, upper]
    return out


def view(file, ident, archived, local, rows, legal, online, name, cuts):
    cv, ols, path, y, valid = local
    move = base.model.movement_prefix(cv, path, rows)
    anchor = np.asarray(archived['score_plan'])
    cap = prefix.saved_plan(move, legal, online, anchor)
    total = int(online.sum())
    if (archived['edits'], archived['cap']) != (total, cap):
        raise ValueError('Original budgets did not reproduce')
    radii = dict(zero=0, one=min(1, total), two=min(2, total), unrestricted=total)

    def measured(plan):
        prefix.saved_plan(move, legal, online, plan)
        p = old.prediction(cv, path, rows, plan)
        a = parent.score(dict(y=y, valid=valid, rec=np.full(len(cv), name)), cv, ols, p, cuts)
        if a['edits'] != total:
            raise ValueError('Re-scored exchange changed the realized edit count')
        return (min(a['vs_cv'], a['vs_ols4']) if a['risk_pass'] else None), a['dual_positive']

    score, dual = measured(anchor)
    if not file.exists():
        terms = old.prefix_terms(cv, ols, path, y, valid, rows, legal, cuts[0])
        pool = []
        for candidate in (anchor, archived['plan']):
            if candidate is not None:
                p = np.asarray(candidate)
                a, _ = measured(p)
                if a is not None:
                    pool.append((solver.exchange_count(p, anchor), a, p.tolist()))
        bounds = {0: dict(upper_pp=score, infeasible=score is None)}
        for distance in sorted(set(radii.values()) - {0}):
            p, bound, status = solver.solve(terms, move, cap, anchor, distance)
            if p is not None:
                a, _ = measured(p)
                if a is not None:
                    pool.append((solver.exchange_count(p, anchor), a, p.tolist()))
            bounds[distance] = dict(upper_pp=bound, infeasible=status == 'infeasible')
        results = {}
        for label, distance in radii.items():
            possible = [b for d, b in bounds.items() if d >= distance]
            infeasible = any(x['infeasible'] for x in possible)
            upper = min((x['upper_pp'] for x in possible if x['upper_pp'] is not None), default=None)
            winner = max((r for r in pool if r[0] <= distance), key=lambda r: r[1], default=None)
            lower = winner[1] if winner else None
            if lower is not None and (infeasible or (upper is not None and lower > upper + 1e-7)):
                raise ValueError('Exclusion disagrees with an original-score witness')
            results[label] = dict(infeasible=infeasible, upper_pp=upper, lower_pp=lower,
                                  plan=winner[2] if winner else None)
        run.once(file, run.stable(dict(identity=ident, edits=total, cap=cap, anchor_dual=dual, results=results)))

    saved = run.read(file)
    if (saved['identity'], saved['edits'], saved['cap'], saved['anchor_dual']) != (ident, total, cap, dual):
        raise ValueError('Resume record changed')
    for name0, r in saved['results'].items():
        if r['plan'] is not None:
            p = np.asarray(r['plan'])
            if solver.exchange_count(p, anchor) > radii[name0] or measured(p)[0] != r['lower_pp']:
                raise ValueError('Saved exchange witness did not verify')
    if archived['plan'] is not None and saved['results']['unrestricted']['lower_pp'] <= FLOORS['positive']:
        raise ValueError('The wider frontier lost an earlier positive witness')
    return saved


def summarize(by_site):
    views = sum(r['views'] for cells in by_site.values() for r in cells)
    summary = {k: {f: [sum(r['bounds'][k][f][i] for cells in by_site.values() for r in cells) for i in range(2)]
                   for f in FLOORS} for k in LEVELS}
    edit_count = sum(r['edits'] for cells in by_site.values() for r in cells)
    anchor = sum(r['anchor_dual'] for cells in by_site.values() for r in cells)
    expected = [summary[k][f] for f in FLOORS for k in LEVELS]
    if any(not 0 <= lo <= hi <= views for lo, hi in expected):
        raise ValueError('Malformed frontier counts')
    for f in FLOORS:
        if any(a > b for i in range(2) for a, b in zip([summary[k][f][i] for k in LEVELS[:-1]],
                                                       [summary[k][f][i] for k in LEVELS[1:]], strict=True)):
            raise ValueError('Frontier is not nested')
    draw = np.random.default_rng(817).integers(len(by_site), size=(3000, len(by_site)))
    x = np.array([np.mean([100.*(r['bounds']['one']['positive'][0] - r['bounds']['zero']['positive'][0])/r['views']
                           for r in group]) for group in by_site.values()])
    return dict(views=views, edits=edit_count, anchor_dual=anchor, bounds=summary,
        verified_one_minus_zero_pp=dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
            locations_positive=int((x > 0.).sum())),
        above_point_one_from_failed_anchor=sum(r['above_point_one_from_failed_anchor'] for cells in by_site.values() for r in cells))


def execute(reg, parent_key, cells, ranks, heads, baseline, previous, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    result = {}
    for site, d in sites.items():
        f = parent.prior.prior.old.source.prior.PRIVATE / f'{site}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            recordings, starts = z['recording'], z['start']
        bank = parent.prior.analytic(d)
        cv, ols = bank[:2]
        q, rec, y, valid = (d[k] for k in ('query', 'rec', 'y', 'valid'))
        cut = np.quantile(np.concatenate([v['cut_samples'] for s, v in sites.items() if s != site]), [.25, .75])
        result[site] = []
        for cell in cells['by_site'][site]:
            seed = cell['seed']
            rank = study.load(ranks['heads'][f'{site}/{seed}/additive_rank'],
                              lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            x, legal, rows, chosen, _ = prefix.menu_for(d, bank, rank)
            head = study.load(heads['count_heads'][f'count_{site}_{seed}_recording'],
                              lambda: study.model.CountHead(x.shape[-1]), study.HEAD_STEPS)
            with torch.inference_mode():
                logits = head(torch.from_numpy(x), torch.from_numpy(legal)).numpy()
            online = study.model.streaming_count(logits, legal, recordings, starts, study.RATE)
            ref = next(r['scores']['recording'] for r in baseline['by_site'][site] if r['seed'] == seed)
            if parent.score(d, cv, ols, old.prediction(cv, chosen, rows, online), cut) != ref:
                raise ValueError('Archived streaming controller did not replay')
            views = []
            for i, name in enumerate(sorted(set(rec))):
                ai, qi = rec == name, recordings == name
                ix, r = np.flatnonzero(ai), rows[qi].copy()
                have = r >= 0
                if (not np.array_equal(np.unique(q[ai]), np.flatnonzero(qi)) or (np.diff(starts[qi]) <= 0).any()
                        or not np.isin(r[have], ix).all()):
                    raise ValueError('Query order or nominee alignment changed')
                r[have] = np.searchsorted(ix, r[have])
                tag = f'{site}_{seed}_{i:04d}.json'
                ident = dict(site=site, seed=seed, ordinal=i)
                a = run.read(base.PRIVATE / tag)
                if a['identity'] != dict(registration=parent_key, **ident):
                    raise ValueError('Unbound archived allocation')
                views.append(view(PRIVATE / tag, dict(registration=reg, **ident), a, (cv[ai], ols[ai], chosen[ai], y[ai], valid[ai]),
                                  r, legal[qi], online[qi], name, cut))
                beat(phase='view', site=site, seed=seed, views=len(views))
            v = dict(seed=seed, views=len(views), edits=sum(r['edits'] for r in views), anchor_dual=sum(r['anchor_dual'] for r in views),
                     bounds={k: intervals([r['results'][k] for r in views]) for k in LEVELS},
                     above_point_one_from_failed_anchor=sum(not r['anchor_dual'] and (r['results']['one']['lower_pp'] or 0.) > .1
                                                           for r in views))
            old_ref = next(r for r in previous['by_site'][site] if r['seed'] == seed)
            if [v[k] for k in ('views', 'edits', 'anchor_dual')] != [old_ref[k] for k in ('views', 'edits', 'score_plan')]:
                raise ValueError('Archived matched allocation did not replay')
            result[site].append(v)
            beat(phase='cell', site=site, seed=seed)
    s = summarize(result)
    if (s['views'], s['edits'], s['anchor_dual']) != (606, 780, 60):
        raise ValueError('Unexpected OpenDD roster')
    return dict(registration=reg, summary=s, by_site=result, original_2pct=True, external_outcome_access=False,
                hindsight_only=True, deployment_changed=False, submission_ready=False)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**data):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **data)
            run.kin.prior.source.atomic(PRIVATE / 'heartbeat.json', row)
            if data['phase'] == 'complete':
                print(row, flush=True)

        r = execute(*register(), beat)
        run.once(OUT / 'readout.json', run.stable(r))
        beat(phase='complete', updates=0, summary=r['summary'])


if __name__ == '__main__':
    main()
