"""Find verified full-grid plans at the archived OpenDD count-and-motion budget."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_full_grid_exchange as prior
from src.evaluation import m3w_full_grid_count_capacity as optimizer

run, before, base, np, torch = prior.run, prior.before, prior.base, prior.np, prior.torch
OUT = prior.OUT.with_name('opendd_full_grid_count_motion_v1')
PRIVATE = prior.PRIVATE.with_name(OUT.name)
VERIFIED = 'verified_full_grid'


def record(inputs, indices, frozen, cap_receipt, path, identity):
    _, _, _, d, bank, menu, _, _, cuts = inputs
    ai, q, rows = indices
    _, legal, _, chosen, _ = menu
    cv, ols = bank[0][ai], bank[1][ai]
    y, valid = d['y'][ai], d['valid'][ai]
    m = base.model.movement_prefix(bank[0], chosen, menu[2])[q]
    legal = legal[q]
    total, cap, status = (frozen[k] for k in ('edits', 'cap', 'status'))
    if frozen['identity']['registration'] != cap_receipt or not np.isfinite(cap) or bool(total) != (cap > 0.):
        raise ValueError('Original registered budget changed')
    ix = np.arange(len(legal))

    def check(count, require_margin=True):
        a = np.asarray(count)
        if (a.shape != ix.shape or not np.issubdtype(a.dtype, np.integer) or a.sum() != total
                or (a < 0).any() or (a > 2).any() or not legal[ix, a].all()
                or not base.model.respects_tokens(a, legal[:, 1]) or not base.model.within_cap(m[ix, a].sum(), cap)):
            raise ValueError('Candidate changed the realized token, count, or movement constraints')
        pred, s = before.measure(d, ai, bank, chosen, rows, a, cuts)
        full, blocked = prior.certificate(cv, pred, y, valid, cuts[0], s)
        good = full and (not require_margin or min(s['vs_cv'], s['vs_ols4']) > optimizer.POSITIVE_PP)
        return s['dual_positive'], good, blocked

    if (frozen['plan'] is not None) != (status == base.PASS):
        raise ValueError('Original feasibility receipt lost its verified witness')
    blind, blind_grid, _ = check(frozen['score_plan'], require_margin=False)
    if not path.exists():
        candidate = None
        reason = ()
        origin = status
        enumerated = 0
        if status == base.PASS:
            original, full, reason = check(frozen['plan'])
            if not original:
                raise ValueError('Original solver witness did not reproduce')
            if full:
                candidate, origin = frozen['plan'], 'reused_witness'
        elif status not in ('zero_realized', 'risk_infeasible', 'no_strict_positive_bound', 'no_support') and not status.startswith('unresolved'):
            raise ValueError('Unexpected original exclusion status')
        if candidate is None and (status == base.PASS or status.startswith('unresolved')):
            if blind_grid and check(frozen['score_plan'])[1]:
                candidate, origin = frozen['score_plan'], 'reused_blind'
            else:
                g, r, e, _, off, easy_cap, mass = base.old.prefix_terms(cv, ols, chosen[ai], y, valid, rows, legal, cuts[0])
                grid = optimizer.prefix_grid_terms(cv, chosen[ai], y, valid, rows, cuts[0])
                candidate, origin = optimizer.solve(legal, m, total, cap, (g, r, e, off, easy_cap, mass), grid)
                if origin == 'unresolved':
                    candidate, origin = optimizer.solve(legal, m, total, cap, (g, r, e, off, easy_cap, mass), grid, seconds=8.)
                if candidate is not None and not check(candidate)[1]:
                    candidate, origin = None, 'unresolved_numerical_check'
                if candidate is None and origin.startswith('unresolved'):
                    plans = optimizer.enumerate_spend(legal, m, total, cap)
                    if plans is not None:
                        origin = 'enumerated_no_witness'
                        for plan in plans:
                            enumerated += 1
                            if check(plan)[1]:
                                candidate, origin = plan, 'enumerated_witness'
                                break
        run.once(path, run.stable(dict(identity=identity, witness=None if candidate is None else list(map(int, candidate)),
                                     robust_status=origin if candidate is None else VERIFIED, origin=origin, enumerated=enumerated,
                                     original_witness_loss=list(reason))))
    saved = run.read(path)
    if saved['identity'] != identity or (saved['robust_status'] == VERIFIED) != (saved['witness'] is not None):
        raise ValueError('Robust count-and-motion checkpoint changed')
    if saved['witness'] is not None and not check(saved['witness'])[1]:
        raise ValueError('Cached robust allocation does not pass both checks')
    full_grid = saved['robust_status'] == VERIFIED
    return dict(views=1, edits=total, archived_positive=int(status == base.PASS), blind_original=int(blind),
                archived_unresolved=int(status.startswith('unresolved')),
                new_original_witness=int(full_grid and status != base.PASS),
                blind_full_grid=int(blind_grid), full_grid=int(full_grid),
                unresolved=int(saved['robust_status'].startswith('unresolved')), enumerated_plans=saved['enumerated'],
                **{f'case_{saved["origin"]}': 1}, **{f'original_loss_{s}': 1 for s in saved['original_witness_loss']})


def compare(groups):
    draws = np.random.default_rng(817).integers(7, size=(3000, 7))
    out = {}
    for l, r in (('full_grid', 'blind_full_grid'), ('full_grid', 'original_verified_after_search'), ('full_grid', 'one_exchange_full_grid')):
        a = np.asarray([np.mean([100*(row[l]-row[r])/row['views'] for row in g]) for g in groups.values()])
        out[f'{l}_minus_{r}'] = dict(point_pp=float(a.mean()), ci95_pp=np.quantile(a[draws].mean(1), [.025, .975]).tolist())
    return out


def execute(beat):
    receipt = base.OUT/'registration.json'
    registration = run.sha(receipt)
    p = base.OUT/'readout.json'
    f = run.read(p)
    one = run.read(prior.OUT/'readout.json')
    if (f['registration'] != registration or f['confirmation_access'] or not f['original_2pct']
            or f['summary']['count_motion'] != 210 or one['summary']['arms']['menu_or_keep']['full_grid'] != 175):
        raise ValueError('Require the archived count-and-motion and one-exchange benchmarks')
    inputs = (Path(__file__), Path(optimizer.__file__), Path(prior.risk.__file__), OUT/'protocol.md', receipt, p,
              prior.OUT/'readout.json', prior.OUT/'registration.json')
    reg = OUT/'registration.json'
    run.once(reg, run.stable(dict(inputs=[run.artifact(x) for x in inputs], original_2pct=True,
                                  no_new_targets=True, solver_seconds=[2., 8.], original_solver_margin_pp=optimizer.POSITIVE_PP,
                                  exact_enumeration_max_plans=25000)))
    bind, groups = run.sha(reg), {}
    for args, _ in prior.matched.context_cells():
        site, seed, _, d, _, menu, _, _, _ = args
        counts = Counter()
        for i, name in enumerate(sorted(set(d['rec']))):
            tag = f'{site}_{seed}_{i:04d}.json'
            pp = base.PRIVATE/tag
            previous = run.read(pp)
            if previous['identity'] != dict(registration=registration, site=site, seed=seed, ordinal=i):
                raise ValueError('Unbound original count-and-motion view')
            identity = dict(registration=bind, original=run.sha(pp))
            counts.update(record(args, before.indices(d, name, menu[2]), previous, registration, PRIVATE/tag, identity))
            beat(phase='recording', site=site, seed=seed, completed=i+1)
        old = next(z for z in f['by_site'][site] if z['seed'] == seed)
        stricter = next(z for z in one['by_location'][site] if z['seed'] == seed)
        if ((counts['views'], counts['edits'], counts['archived_positive'], counts['blind_original'])
                != (old['views'], old['edits'], old['count_motion'], old['score_plan'])
                or counts['blind_full_grid'] != stricter['outcomes']['incumbent']['full_grid']):
            raise ValueError('Original fixed-spend reference did not reproduce')
        counts['one_exchange_full_grid'] = int(stricter['outcomes']['menu_or_keep']['full_grid'])
        counts['original_verified_after_search'] = counts['archived_positive']+counts['new_original_witness']
        if counts['new_original_witness'] > counts['archived_unresolved']:
            raise ValueError('A full-grid witness contradicts an original exclusion')
        groups.setdefault(site, []).append(dict(seed=seed, **counts))
        beat(phase='cell', site=site, seed=seed, full_grid=counts['full_grid'], unresolved=counts['unresolved'])
    if len(groups) != 7 or any({r['seed'] for r in g} != {17, 29, 43} for g in groups.values()):
        raise ValueError('Expected all original sources and seeds')
    counts = sum((Counter({k: v for k, v in r.items() if k != 'seed'}) for g in groups.values() for r in g), Counter())
    counts.setdefault('unresolved', 0)
    counts['original_unresolved_after_search'] = counts['archived_unresolved']-counts['new_original_witness']
    counts['verified_upper_including_unresolved'] = counts['full_grid']+counts['unresolved']
    result = dict(registration=bind, summary=dict(counts), by_location=groups, contrasts=compare(groups),
                  bootstrap_draws=3000, new_external_outcome_access=False, training=False,
                  original_2pct=True, hindsight_only=True, deployment_changed=False, submission_ready=False)
    run.once(OUT/'readout.json', run.stable(result))
    print(result['summary'], result['contrasts'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**event):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **event)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            if event['phase'] == 'cell':
                print(row, flush=True)

        execute(beat)
