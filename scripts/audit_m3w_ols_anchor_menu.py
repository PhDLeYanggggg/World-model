"""Measure path-bank capacity with exact counts and the original observed risk."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_fixed_budget_gain as previous
from scripts import audit_m3w_cv_joint_feasibility as audit
from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.world_model import m3w_ols_anchor_paths as model

old, base, prior, np = previous.old, previous.base, previous.prior, previous.np
OUT = previous.OUT.parent / 'moving_ols_anchor_menu_v2'
PRIVATE = previous.PRIVATE.parent / OUT.name
SECONDS = 3.
ARMS = ('old', 'expanded')
m = audit.model


def inputs(cv, ols, bank, y, valid, query, active, anchor, cut):
    d = m.coefficients(cv[active], ols[active], bank, y[active], valid[active], query[active], cut)
    count = np.bincount(d['group'][anchor[active] >= 0], minlength=len(d['counts']))
    ref, offset, _, _ = candidate_terms(cv, ols, cv[:, None], y, valid)
    w = old.weight(query)
    ade, _ = m.errors(cv, y, valid)
    ref = float(w @ ref)
    if ref > 0.:
        d['gain'] *= d['ref']/ref
        d['offset'] = np.array([0., w @ offset])*(100./ref)
    d['ref'] = ref
    d['easy_cap'] = .02*float(ade[valid.any(1) & (ade > 0.) & (ade <= cut)].sum())
    return d, count


def solve(d, count):
    if d['ref'] <= 0. or not d['easy'].any():
        return None, dict(status='undefined_support', upper_pp=None)
    n = len(d['row'])
    mat, rhs = m.constraints(d, int(count.max()))
    equal = m.coo_array((np.ones(n), (d['group'][d['row']], np.arange(n))),
                        shape=(len(count), n+1)).tocsc()
    c = np.r_[np.zeros(n), -1.]
    lp = m.linprog(c, A_ub=mat, b_ub=rhs, A_eq=equal, b_eq=count,
                   bounds=[(0., 1.)]*n+[(None, None)], method='highs',
                   options={'time_limit': SECONDS})
    upper = float(-lp.fun) if lp.status == 0 else None
    log = dict(status='unresolved', upper_pp=upper, lp=int(lp.status))
    if lp.status == 2:
        return None, dict(log, status='risk_infeasible')
    if upper is not None and upper < -m.POSITIVE_PP:
        return None, dict(log, status='nonpositive_bound')
    if lp.status == 0 and np.max(np.abs(lp.x[:-1]-np.rint(lp.x[:-1]))) <= 1e-8:
        return lp.x[:-1] > .5, log
    mip = m.milp(c, integrality=np.r_[np.ones(n), 0.],
        bounds=m.Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=[m.LinearConstraint(mat, -np.inf, rhs), m.LinearConstraint(equal, count, count)],
        options={'time_limit': SECONDS, 'mip_rel_gap': 0., 'presolve': False})
    bound = getattr(mip, 'mip_dual_bound', None)
    if bound is not None and np.isfinite(bound):
        upper = min(upper, -float(bound)) if upper is not None else -float(bound)
    log.update(upper_pp=upper, mip=int(mip.status))
    if mip.status == 2:
        return None, dict(log, status='risk_infeasible')
    if upper is not None and upper < -m.POSITIVE_PP:
        return None, dict(log, status='nonpositive_bound')
    return (mip.x[:-1] > .5 if mip.x is not None else None), log


def receipt(actions, cv, ols, bank, y, valid, query, active, anchor, cut):
    row, arm = np.asarray(actions, dtype=int).reshape(-1, 2).T
    if (row < 0).any() or (row >= len(active)).any() or (arm < 0).any() or (arm >= bank.shape[1]).any():
        raise ValueError('Action outside the candidate bank')
    take = active[row]
    if len(np.unique(take)) != len(take):
        raise ValueError('Duplicate action recipient')
    path = cv.copy()
    path[take] = bank[row, arm]
    ids, q = np.unique(query, return_inverse=True)
    count = np.bincount(q[take], minlength=len(ids))
    expected = np.bincount(q[anchor >= 0], minlength=len(ids))
    r = audit.policy(path, cv, cv, y, valid, np.repeat('r', len(cv)),
                     easy_cut=cut[0], hard_cut=cut[1])
    risk = r['recordings']['r']
    w = old.weight(query)
    terms = [audit.motion.full_masses(x, path, cv, y, valid, w) for x in (cv, ols)]
    ref = terms[0]['reference']
    gain = [100*(v['known_delta']-v['unknown_radius'])/ref for v in terms] if ref > 0. else None
    return old.stable(dict(gain_pp=gain, risk=risk,
        verified=bool(np.array_equal(count, expected) and risk['supported'] and risk['selected'] == len(take)
                      and gain is not None and min(gain) > m.POSITIVE_PP)))


def search(cv, ols, bank, y, valid, q, active, anchor, cut, inherited=None):
    if not len(active):
        return dict(status='no_edits', actions=None, solver=None, receipt=None, inherited=False)
    d, count = inputs(cv, ols, bank, y, valid, q, active, anchor, cut[0])
    chosen, log = solve(d, count)
    args = (cv, ols, bank, y, valid, q, active, anchor, cut)
    actions = np.column_stack((d['row'][chosen], d['arm'][chosen])).tolist() if chosen is not None else None
    r = receipt(actions, *args) if actions is not None else None
    status = 'verified' if r is not None and r['verified'] else log['status']
    reused = False
    if inherited is not None and status != 'verified':
        if status != 'unresolved':
            raise ValueError('Solver result contradicts a known nested witness')
        r = receipt(inherited, *args)
        if not r['verified']:
            raise ValueError('Nested witness changed')
        actions, status, reused = inherited, 'verified', True
    return dict(status=status, actions=actions, solver=log, receipt=r, inherited=reused)


def summarize(rows, old_arms):
    statuses = {k: dict(Counter(r[k]['status'] for r in rows)) for k in ARMS}
    transitions = Counter(r['old']['status']+' -> '+r['expanded']['status'] for r in rows)
    sites = sorted({r['site'] for r in rows})
    rates = np.array([np.mean([100.*(int(r['expanded']['status'] == 'verified')
                                     - int(r['old']['status'] == 'verified'))
                               for r in rows if r['site'] == s]) for s in sites])
    rng = np.random.default_rng(73107)
    draw = rng.integers(len(rates), size=(3000, len(rates)))
    selected = Counter(model.NAMES[arm-old_arms] for r in rows if r['expanded']['status'] == 'verified'
                       for _, arm in r['expanded']['actions'] if arm >= old_arms)
    return dict(statuses=statuses, transitions=dict(transitions), new_path_choices=dict(selected),
                views=len(rows), edits=sum(r['edits'] for r in rows),
                effect_pp=float(rates.mean()), ci95_pp=np.quantile(rates[draw].mean(1), [.025, .975]).tolist(),
                source_groups=len(sites), bootstrap_draws=3000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay', action='store_true')
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source, freeze = [previous.OUT / name for name in ('readout.json', 'freeze.json')]
    r, f = map(old.read, (source, freeze))
    if (r['freeze'] != old.sha(freeze) or not f['before_held_scoring'] or not r['original_2pct']
            or r['a_labels_opened'] or r['independent_confirmation']):
        raise ValueError('Require frozen development decisions')
    reg = OUT / 'registration.json'
    old.once(reg, old.stable(dict(parents=[old.artifact(x) for x in (source, freeze)],
        files=[old.artifact(x) for x in (Path(__file__), Path(model.__file__),
               Path(m.__file__), Path(audit.policy.__code__.co_filename), OUT / 'protocol.md')],
        paths=model.NAMES, seconds=SECONDS, mip_presolve=False, original_2pct=True, exact_query_counts=True,
        superseded=old.artifact(OUT.parent / 'moving_ols_anchor_menu_v1' / 'registration.json'),
        outcome_aware_diagnostic_only=True, independent_confirmation=False, a_labels_opened=False)))
    with (PRIVATE / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.register()
        data = base.fit_population(lambda **_: None)
        rows, width = [], None
        for i, c in enumerate(f['cells']):
            ident = c['identity']
            _, h = base.subset(data, ident['held'])
            with np.load(old.checked(c['choices']), allow_pickle=False) as z:
                np.testing.assert_array_equal(h['ids'], z['ids'])
                anchor = z['anchor']
            for record in np.unique(h['recording']):
                take = h['recording'] == record
                q, history, y, v = (h[k][take] for k in ('query', 'history', 'target', 'valid'))
                a = anchor[take]
                active = np.flatnonzero(np.isin(q, q[a >= 0]))
                cv, ols = None, None
                old_bank = np.empty((0, 0, 12, 2))
                if len(active):
                    # Generate the menu only where the frozen count permits changes.
                    b = prior.study.motion.bank(history)
                    cv, ols = b[:, 0], b[:, 1]
                    ray = base.expanded(cv[active], b[active, 1:])
                    old_bank = np.stack([previous.rollout(cv[active], ray, a)
                         for a in range(ray.shape[1]*previous.amplitude.M)], axis=1)
                    width = old_bank.shape[1]
                new_bank = (np.concatenate((old_bank, model.candidates(history[active])), axis=1)
                            if len(active) else old_bank)
                file = PRIVATE / f'view_{i}_{record}.json'
                key = dict(registration=old.sha(reg), cell=i, recording=int(record))
                if not file.exists():
                    if args.replay:
                        raise ValueError('Missing capacity checkpoint')
                    old_result = search(cv, ols, old_bank, y, v, q, active, a, c['cut'])
                    inherited = old_result['actions'] if old_result['status'] == 'verified' else None
                    new_result = search(cv, ols, new_bank, y, v, q, active, a, c['cut'], inherited)
                    old.once(file, old.stable(dict(key=key, old=old_result, expanded=new_result)))
                saved = old.read(file)
                if saved['key'] != key:
                    raise ValueError('Capacity checkpoint binding mismatch')
                for arm, bank in zip(ARMS, (old_bank, new_bank), strict=True):
                    r0 = saved[arm]
                    if r0['actions'] is not None and receipt(
                            r0['actions'], cv, ols, bank, y, v, q, active, a, c['cut']) != r0['receipt']:
                        raise ValueError('Capacity witness did not replay')
                site, = np.unique(h['sites'][take])
                rows.append(dict(saved, site=str(site), seed=ident['seed'], edits=int((a >= 0).sum())))
                prior.donor.prior.prior.source.atomic(PRIVATE / 'heartbeat.json',
                    dict(pid=os.getpid(), cell=i, recording=int(record), views=len(rows)))
            print(dict(pid=os.getpid(), cell=i, views=len(rows)), flush=True)
        summary = summarize(rows, width)
        if summary['views'] != 72 or summary['edits'] != r['edits']['anchor']:
            raise ValueError('Changed development roster or count')
        public = [{**{k: x[k] for k in ('key', 'site', 'seed', 'edits')},
                   **{k: {a: b for a, b in x[k].items() if a != 'actions'} for k in ARMS}} for x in rows]
        old.once(OUT / 'readout.json', old.stable(dict(registration=old.artifact(reg), summary=summary, views=public,
            original_2pct=True, independent_confirmation=False, a_labels_opened=False)))
        print(summary, flush=True)


if __name__ == '__main__':
    main()
