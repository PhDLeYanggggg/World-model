"""Compare frozen OpenDD decisions to a hindsight solver at the same spent counts."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_streaming_menu_capacity as broad
from src.evaluation import m3w_realized_stream_capacity as exact


study, run, np, torch, parent = broad.study, broad.run, broad.np, broad.torch, broad.parent
OUT = broad.OUT.parent/'opendd_realized_stream_capacity_v1'
PRIVATE = broad.PRIVATE.parent/OUT.name
PASS = broad.POS


def register():
    old, earlier, ranks = broad.register()
    inputs = [broad.OUT/'analysis.json', broad.OUT/'registration.json', study.OUT/'analysis.json',
              study.OUT/'head_freeze.json', OUT/'protocol.md', Path(__file__), Path(exact.__file__)]
    cap, counts, heads = (run.read(inputs[i]) for i in (0, 2, 3))
    if (cap['registration'] != old or counts['heads'] != run.sha(inputs[3])
            or any(r['confirmation_access'] or not r['original_2pct'] for r in (cap, counts))):
        raise ValueError('Require the verified development-only parent scores')
    path = OUT/'registration.json'
    run.once(path, run.stable(dict(inputs=[run.artifact(p) for p in inputs], seconds_per_milp=2.,
        observed_forecast_steps=[8, 12], count_constraint='equal_online_total_per_recording_and_seed',
        token_credit=.2, token_cap=2., original_2pct=True, hindsight_only=True, no_fitting=True,
        confirmation_access=False)))
    return run.sha(path), old, earlier, ranks, heads, counts, cap


def prefix_terms(cv, ols, path, target, valid, rows, legal, cut):
    m = broad.solver.coefficients(cv, ols, path[:, None], target, valid, cut)
    w = np.full(len(cv), -1, dtype=int)
    w[m['row']] = np.arange(len(m['row']))
    nominated = rows >= 0
    ix = w[rows[nominated]]
    if (ix < 0).any() or len(ix) != len(m['row']) or len(set(ix)) != len(ix):
        raise ValueError('Frozen nominee roster does not match realized nonidentity paths')
    g = np.zeros((*rows.shape, 2))
    r = np.zeros((*rows.shape, 3))
    support = np.zeros(rows.shape, dtype=bool)
    g[nominated], r[nominated], support[nominated] = m['gain'][:, ix].T, m['risk'][:, ix].T, m['easy'][ix]
    pref = lambda v: np.concatenate((np.zeros_like(v[:, :1]), np.cumsum(v, axis=1)), axis=1)
    return (pref(g), pref(r), pref(support) > 0, legal, m['offset'], m['easy_cap'], m['cv_mass'])


def prediction(cv, path, rows, counts):
    pred = cv.copy()
    kept = rows[np.arange(2) < counts[:, None]]
    if (kept < 0).any() or len(kept) != len(set(kept)):
        raise ValueError('Illegal or duplicate nominated agent')
    pred[kept] = path[kept]
    return pred


def solve_view(identity, file, term, cv, ols, path, rows, online, target, valid, name, cuts, upper_status):
    total = int(online.sum())
    legal = term[3]

    def checked(count):
        count = np.asarray(count)
        if (count.shape != online.shape or not np.issubdtype(count.dtype, np.integer) or (count < 0).any()
                or (count > 2).any() or count.sum() != total or not legal[np.arange(len(count)), count].all()
                or not broad.solver.respects_tokens(count, legal[:, 1])):
            raise ValueError('Cached plan is not the same realized budget')
        pred = prediction(cv, path, rows, count)
        if (pred != cv).any((1, 2)).sum() != total:
            raise ValueError('Realized edit count changed on re-score')
        r = parent.score(dict(y=target, valid=valid, rec=np.full(len(cv), name)), cv, ols, pred, cuts)
        return bool(r['dual_positive'])

    actual = checked(online)
    if not file.exists():
        c, result = exact.solve(*term, total)
        if c is not None and not checked(c):
            c, result = None, dict(status='unresolved_numerical_check', upper_pp=result['upper_pp'])
        if c is None and actual:
            if result['status'] not in ('unresolved', 'unresolved_numerical_check'):
                raise ValueError('Solver exclusion contradicts the verified online witness')
            c = online
        if c is not None:
            result['status'] = PASS
        run.once(file, run.stable(dict(identity=identity, total=total, actual=actual, result=result,
                                      plan=c.tolist() if c is not None else None)))
    saved = run.read(file)
    if (saved['identity'], saved['total'], saved['actual']) != (identity, total, actual):
        raise ValueError('The frozen realized-count inputs changed')
    status = saved['result']['status']
    if status == PASS:
        if not checked(saved['plan']) or upper_status not in (PASS, 'unresolved'):
            raise ValueError('Fixed-count witness disagrees with the original risk rule or broader quota')
    elif actual or saved['plan'] is not None:
        raise ValueError('Cached result lost a verified online witness')
    return status, actual, total


def execute(key, previous, earlier, ranks, heads, frozen_counts, cap, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    summary, by_site = Counter(), {}
    for site, data in sites.items():
        src = parent.prior.prior.old.source.prior.PRIVATE/f'{site}_cohort.npz'
        with np.load(src, allow_pickle=False) as z:
            recordings, starts = z['recording'], z['start']
        bank = parent.prior.analytic(data)
        cv, ols, paths = bank[:3]
        q, agent_rec, target, valid = (data[k] for k in ('query', 'rec', 'y', 'valid'))
        cuts = np.quantile(np.concatenate([v['cut_samples'] for s, v in sites.items() if s != site]), [.25, .75])
        actual_rows = {r['seed']: r['scores']['recording'] for r in frozen_counts['by_site'][site]}
        by_seed = []
        for e in earlier['by_site'][site]:
            seed = e['seed']
            rank = study.load(ranks['heads'][f'{site}/{seed}/additive_rank'],
                              lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            menu = study.menus(data, bank, rank)
            net = study.load(heads['count_heads'][f'count_{site}_{seed}_recording'],
                             lambda: study.model.CountHead(menu['x'].shape[-1]), study.HEAD_STEPS)
            legal, rows, arms = (menu[k] for k in ('legal', 'row', 'arm'))
            with torch.inference_mode():
                logits = net(torch.from_numpy(menu['x']), torch.from_numpy(legal)).numpy()
            online = study.model.streaming_count(logits, legal, recordings, starts, study.RATE)
            at = rows >= 0
            chosen = cv.copy()
            chosen[rows[at]] = paths[rows[at], arms[at]]
            pred = prediction(cv, chosen, rows, online)
            if run.stable(parent.score(data, cv, ols, pred, cuts)) != actual_rows[seed]:
                raise ValueError('Archived controller paths or outcomes did not reproduce')
            out = Counter()
            for i, name in enumerate(sorted(set(agent_rec))):
                ai = agent_rec == name
                qi = recordings == name
                local_rows = rows[qi].copy()
                nonzero = local_rows >= 0
                indices = np.flatnonzero(ai)
                if (not np.array_equal(np.flatnonzero(qi), np.unique(q[ai]))
                        or not np.isin(local_rows[nonzero], indices).all()):
                    raise ValueError('Queries or nominated actors do not align with the recording')
                local_rows[nonzero] = np.searchsorted(indices, local_rows[nonzero])
                old = run.read(broad.PRIVATE/f'{site}_{i:04d}.json')
                if (old['registration'], old['site'], old['ordinal']) != (previous, site, i):
                    raise ValueError('The quota-level recording receipt changed')
                earlier_status = next(v['arms']['nested_stream']['status'] for v in old['entries'] if v['seed'] == seed)
                local = (cv[ai], ols[ai], chosen[ai], target[ai], valid[ai])
                terms = prefix_terms(*local, local_rows, legal[qi], cuts[0])
                ident = dict(registration=key, site=site, ordinal=i, seed=seed)
                status, good, spend = solve_view(ident, PRIVATE/f'{site}_{seed}_{i:04d}.json', terms,
                    *local[:3], local_rows, online[qi], *local[3:], name, cuts, earlier_status)
                out.update(views=1, edits=spend, online_pass=good, exact_pass=status == PASS,
                           broad_pass=earlier_status == PASS, **{status: 1})
                beat(phase='recording', site=site, seed=seed, recordings=i+1)
            if out['online_pass'] != actual_rows[seed]['dual_positive'] or out['edits'] != actual_rows[seed]['edits']:
                raise ValueError('Record-level decomposition did not reproduce the archived score')
            by_seed.append(dict(seed=seed, **out))
            summary.update(out)
            beat(phase='site_seed_complete', site=site, seed=seed, counts=dict(out))
        by_site[site] = by_seed
    if (summary['views'], summary['edits'], summary['online_pass'], summary['broad_pass']) != (606, 780, 57, 330):
        raise ValueError('The original roster or edit budget changed')
    return dict(registration=key, summary=dict(summary), by_site=by_site,
                original_2pct=True, confirmation_access=False, hindsight_only=True,
                no_fitting=True, deployment_changed=False, submission_ready=False)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-started, 3), **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            if v['phase'] != 'recording':
                print(row, flush=True)

        result = execute(*register(), beat)
        run.once(OUT/'readout.json', run.stable(result))
        beat(phase='complete', summary=result['summary'], updates=0)


if __name__ == '__main__':
    main()
