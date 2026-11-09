"""Compare query-fixed and recording-pooled Moving development budgets."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_feasibility as check
from scripts import run_m3w_moving_dual_record_forecast as previous
from src.evaluation import m3w_record_peak_allocation as model

old, np, torch = previous.old, previous.np, previous.torch
OUT = previous.OUT.parent / 'moving_record_peak_allocation_v1'
PRIVATE = previous.PRIVATE.parent / OUT.name
BASE = ('ols4', 'ols8', 'completion', 'eqmotion', 'pct')
ADDED = ('ols_only', 'dual_risk', 'diagnostic')
FRACTIONS = (.10, .35, .70, 1.)
OPTIONS = ('base_query', 'base_record', 'expanded_query', 'expanded_record')
SECONDS = 3.


def register():
    key, *_ = previous.register()
    fp, rp = (previous.OUT / name for name in ('freeze.json', 'readout.json'))
    f, r = map(old.read, (fp, rp))
    if (f['registration'] != key or r['freeze'] != old.sha(fp) or not f['before_held_scoring']
            or not r['original_2pct'] or any(r[k] for k in ('a_labels_opened', 'confirmation_access', 'independent_confirmation'))):
        raise ValueError('Only frozen development inputs are permitted')
    p = OUT / 'registration.json'
    old.once(p, old.stable(dict(parents=[old.artifact(z) for z in (fp, rp)],
        code=[old.artifact(z) for z in (Path(__file__), Path(model.__file__), Path(check.__file__), OUT / 'protocol.md')],
        base=['half_anchor', *BASE], added=ADDED, fractions=FRACTIONS, seconds_per_lp_and_mip=SECONDS,
        modes=OPTIONS, original_2pct=True, a_labels_opened=False, confirmation_access=False)))
    return old.sha(p), f['cells'], r


def budgets(cv, anchor, query):
    _, group = np.unique(query, return_inverse=True)
    radius = np.linalg.norm(anchor-cv, axis=-1).max(1)
    n = int(group.max()) + 1
    return group, np.bincount(group[radius > 0.], minlength=n), np.bincount(group, weights=radius, minlength=n)


def verify(actions, args, mode, group, edits, radii, bound):
    cv, _, bank, _, _, query, *_ = args
    actions = np.asarray(actions, dtype=int).reshape(-1, 2)
    receipt = check.verify(actions, *args, cap=len(query))
    row, arm = actions.T
    peak = np.linalg.norm(bank[row, arm] - cv[row], axis=-1).max(1)
    if mode == 'record':
        g, n, r = np.zeros(len(row), dtype=int), np.array([edits.sum()]), np.array([radii.sum()])
    else:
        g, n, r = group[row], edits, radii
    count = np.bincount(g, minlength=len(n))
    used = np.bincount(g, weights=peak, minlength=len(n))
    matched = (count <= n).all() and (used <= r + 1e-8 * (1. + r)).all()
    if receipt['gain_pp'] is not None and np.any(np.asarray(receipt['gain_pp']) > np.asarray(bound) + 1e-6):
        raise ValueError('Finite path exceeds arbitrary-path envelope')
    return dict(verified=bool(matched and receipt['verified']), risk=receipt['original_risk'],
                gains=receipt['gain_pp'], edits=len(row), peak=float(used.sum()), active_queries=len(set(group[row])))


def optimize(args, mode, group, edits, radii, bound, inherited):
    cv, ols, bank, y, v, q, easy, _ = args
    if bound is None:
        return dict(status='undefined_support', solver=None, actions=None)
    if min(bound) < -model.POSITIVE_PP:
        if inherited:
            raise ValueError('Nested witness contradicts arbitrary bound')
        return dict(status='arbitrary_excluded', solver=None, actions=None)
    d = model.design(cv, ols, bank, y, v, q, easy, edits, radii, mode)
    picked, solver = model.solve(d, SECONDS)
    solver.update(actions_before_prune=d['before_prune'], actions_after_prune=len(d['row']))
    candidate = np.column_stack((d['row'][picked], d['arm'][picked])).tolist() if picked is not None else None
    choices = [a for a in [candidate, *inherited] if a]
    valid = []
    for a in choices:
        z = verify(a, args, mode, group, edits, radii, bound)
        if z['verified']:
            valid.append((min(z['gains']), a, z))
    if valid:
        if solver['status'] != 'unresolved':
            raise ValueError('Nested witness contradicts finite solver exclusion')
        _, a, z = max(valid, key=lambda v: v[0])
        return dict(status='verified', solver=solver, actions=a, witness=z)
    if inherited:
        raise ValueError('Nested feasible set lost an exact witness')
    return dict(status=solver['status'], solver=solver, actions=None)


def solve_record(path, ident, cv, ols, anchor, bank, width, y, valid, q, cut, replay):
    group, edits, radii = budgets(cv, anchor, q)
    bounds = model.arbitrary_bound(cv, ols, y, valid, q, radii)
    anchor_args = (cv, ols, anchor[:, None], y, valid, q, *cut)
    anchor_actions = np.column_stack((np.flatnonzero((anchor != cv).any((1, 2))), np.zeros(edits.sum(), dtype=int))).tolist()
    starting = verify(anchor_actions, anchor_args, 'query', group, edits, radii, bounds['query'])
    if not path.exists():
        if replay:
            raise ValueError('Missing capacity checkpoint')
        done = {}
        for name in OPTIONS:
            menu, mode = name.split('_')
            args = (cv, ols, bank[:, :width] if menu == 'base' else bank, y, valid, q, *cut)
            keys = (f'{menu}_query',) if mode == 'record' else ()
            if menu == 'expanded':
                keys += (f'base_{mode}',)
            inherited = [done[k]['actions'] for k in keys if done[k]['status'] == 'verified']
            if starting['verified']:
                inherited.append(anchor_actions)
            done[name] = optimize(args, mode, group, edits, radii, bounds[mode], inherited)
        old.once(path, old.stable(dict(identity=ident, arbitrary=bounds, starting=starting, plans=done,
            budget=dict(edits=int(edits.sum()), peak=float(radii.sum()), queries=int((edits > 0).sum())))))
    z = old.read(path)
    if z['identity'] != ident or z['arbitrary'] != bounds or z['starting'] != starting:
        raise ValueError('Capacity checkpoint inputs changed')
    for name, r in z['plans'].items():
        if r['status'] != 'verified':
            continue
        menu, mode = name.split('_')
        args = (cv, ols, bank[:, :width] if menu == 'base' else bank, y, valid, q, *cut)
        receipt = verify(r['actions'], args, mode, group, edits, radii, bounds[mode])
        if not receipt['verified'] or receipt != r['witness']:
            raise ValueError('Witness failed replay of budgets, support or utility')
    return dict(identity=ident, arbitrary=bounds, starting=starting, budget=z['budget'],
                plans={k: {a: b for a, b in v.items() if a != 'actions'} for k, v in z['plans'].items()})


def summarize(rows):
    sites = sorted({r['identity']['source'] for r in rows})
    seeds = {r['identity']['seed'] for r in rows}
    ids = {(r['identity']['recording'], r['identity']['seed']) for r in rows}
    if len(rows) != 72 or len(ids) != 72 or len(sites) != 9 or len(seeds) != 3:
        raise ValueError('Expected 24 recordings times three seeds over nine sources')
    draws = np.random.default_rng(40219).integers(len(sites), size=(3000, len(sites)))
    verified = lambda r, k: int(r['plans'][k]['status'] == 'verified')
    stats = {}
    for a, b in (('base_query', 'base_record'), ('expanded_query', 'expanded_record'),
                 ('base_query', 'expanded_query'), ('base_record', 'expanded_record')):
        delta = np.array([np.mean([100. * (verified(r, b) - verified(r, a))
                          for r in rows if r['identity']['source'] == s]) for s in sites])
        if (delta < 0.).any():
            raise ValueError('Nested budget/menu lost verified views')
        stats[f'{b}_minus_{a}'] = dict(point_pp=float(delta.mean()),
             ci95_pp=np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
             added_verified=sum(verified(r, b) - verified(r, a) for r in rows))
    arbitrary = {}
    for k in ('query', 'record'):
        v = np.array([r['arbitrary'][k] for r in rows])
        arbitrary[k] = dict(excluded=int((v.min(1) < -model.POSITIVE_PP).sum()),
            ols4_excluded=int((v[:, 1] < -model.POSITIVE_PP).sum()), median_pp=np.median(v, axis=0).tolist())
    return dict(views=len(rows), recordings=len({rec for rec, _ in ids}), sources=len(sites),
        anchor_dual=sum(r['starting']['verified'] for r in rows), arbitrary=arbitrary,
        status={k: dict(Counter(r['plans'][k]['status'] for r in rows)) for k in OPTIONS},
        paired_verified_rates=stats, bootstrap_draws=len(draws), bootstrap_unit='source_after_seed_average')


def execute(reg, cells, replay, beat):
    base, prior = previous.base, previous.prior
    data = base.fit_population(beat)
    linear = prior.study.motion.bank(data['history'])[:, :3]
    rows = []
    for c in cells:
        i, cut = c['identity'], c['cut']
        ix, h = base.subset(data, i['held'])
        cv, ols, ols8 = linear[ix].transpose(1, 0, 2, 3)
        with np.load(old.checked(c['paths']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            anchor = z['half_anchor']
            paths = dict(ols4=ols, ols8=ols8, **{k: z[k] for k in (*BASE[2:], *ADDED)})
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            s, = np.unique(h['sites'][at])
            ident = dict(registration=reg, fold=i['fold'], seed=i['seed'], source=str(s), recording=str(rec))
            c0, a0 = cv[at], anchor[at]
            bank = np.stack([a0, *(c0 + a * (paths[k][at] - c0) for k in (*BASE, *ADDED) for a in FRACTIONS)], axis=1)
            y, v, q = (h[k][at] for k in ('target', 'valid', 'query'))
            path = PRIVATE / f"view_{i['fold']}_{i['seed']}_{int(rec)}.json"
            z = solve_record(path, ident, c0, ols[at], a0, bank, 1 + len(BASE)*len(FRACTIONS), y, v, q, cut, replay)
            rows.append(z)
            beat(phase='record_complete', completed=len(rows), total=72,
                 statuses={k: t['status'] for k, t in z['plans'].items()})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    replay = p.parse_args().replay
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / 'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**z):
            v = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), replay=replay, **z)
            previous.prior.donor.prior.prior.source.atomic(PRIVATE / 'heartbeat.json', v)
            print(v, flush=True)
        reg, cells, r = register()
        rows = execute(reg, cells, replay, beat)
        result = summarize(rows)
        if result['anchor_dual'] != r['dual_pass']['half_anchor']:
            raise ValueError('Anchor pass count changed')
        old.once(OUT / 'readout.json', old.stable(dict(registration=reg, rows=rows, summary=result,
            original_2pct=True, hindsight_only=True, new_training=False, a_labels_opened=False,
            confirmation_access=False, independent_confirmation=False, deployment_changed=False)))
        beat(phase='complete', summary=result)


if __name__ == '__main__':
    main()
