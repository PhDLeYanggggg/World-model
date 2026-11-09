"""Search a small integer neighborhood for the unresolved Moving witnesses."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_record_peak_allocation as audit

m, np, old = audit.model, audit.np, audit.old


def kernel(d, nearby=32, seconds=8.):
    a, b = m.constraints(d)
    n = len(d['row'])
    c = np.r_[np.zeros(n), -1.]
    lp = m.linprog(c, A_ub=a, b_ub=b, bounds=[(0., 1.)]*n+[(None, None)], method='highs',
                  options={'time_limit': seconds})
    log = dict(lp=int(lp.status), fixed=None, free=None, mip=None)
    if lp.status != 0 or -lp.fun <= m.POSITIVE_PP:
        return None, log
    frac = (lp.x[:n] > 1e-8) & (lp.x[:n] < 1.-1e-8)
    near_rows = set(d['row'][frac])
    reduced = np.maximum(lp.lower.marginals[:n], -lp.upper.marginals[:n])
    added = 0
    for j in np.argsort(reduced, kind='stable'):
        row = d['row'][j]
        if row not in near_rows:
            near_rows.add(row)
            added += 1
            if added == nearby:
                break
    free = np.isin(d['row'], list(near_rows))
    fixed = (lp.x[:n] >= 1.-1e-8) & ~free
    z = np.r_[free, True]
    rhs = b - a @ np.r_[fixed, 0.]
    k = int(free.sum())
    res = m.milp(c[z], integrality=np.r_[np.ones(k), 0.],
        bounds=m.Bounds(np.r_[np.zeros(k), -np.inf], np.r_[np.ones(k), np.inf]),
        constraints=m.LinearConstraint(a[:, z], -np.inf, rhs),
        options={'time_limit': seconds, 'mip_rel_gap': 0.})
    log.update(fixed=int(fixed.sum()), free=k, mip=int(res.status), fractional=int(frac.sum()))
    if res.x is None:
        return None, log
    fixed[free] = res.x[:-1] > .5
    return np.column_stack((d['row'][fixed], d['arm'][fixed])).tolist(), log


def main():
    reg, cells, _ = audit.register()
    rp = audit.OUT / 'readout.json'
    p = old.read(rp)
    if (p['registration'] != reg or not p['original_2pct'] or not p['hindsight_only']
            or any(p[k] for k in ('a_labels_opened', 'confirmation_access', 'independent_confirmation'))):
        raise ValueError('Require verified development-only allocation')
    seconds, nearby = 8., 32
    binding = dict(parent=old.artifact(rp), code=old.artifact(Path(__file__)), seconds=seconds, nearby_rows=nearby)
    by_cell = {}
    for row in p['rows']:
        if any(z['status'] == 'unresolved' for z in row['plans'].values()):
            i = row['identity']
            by_cell.setdefault((i['fold'], i['seed']), []).append(row)
    base, prior = audit.previous.base, audit.previous.prior
    d = base.fit_population(lambda **_: None)
    linear = prior.study.motion.bank(d['history'])[:, :3]
    rows = []
    for cell in cells:
        i, cut = cell['identity'], cell['cut']
        todo = by_cell.get((i['fold'], i['seed']), [])
        if not todo:
            continue
        ix, h = base.subset(d, i['held'])
        cv, ols, ols8 = linear[ix].transpose(1, 0, 2, 3)
        with np.load(old.checked(cell['paths']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            anchor = z['half_anchor']
            paths = dict(ols4=ols, ols8=ols8, **{k: z[k] for k in (*audit.BASE[2:], *audit.ADDED)})
        for row in todo:
            ident = row['identity']
            rec = ident['recording']
            at = h['recording'] == int(rec)
            c0, a0 = cv[at], anchor[at]
            bank = np.stack([a0, *(c0 + a * (paths[k][at] - c0) for k in (*audit.BASE, *audit.ADDED)
                                  for a in audit.FRACTIONS)], axis=1)
            width = 1 + len(audit.BASE)*len(audit.FRACTIONS)
            y, v, q = (h[k][at] for k in ('target', 'valid', 'query'))
            group, edits, radii = audit.budgets(c0, a0, q)
            cp = audit.PRIVATE / f"repair_{i['fold']}_{i['seed']}_{rec}.json"
            solved = {}
            saved = old.read(cp) if cp.exists() else None
            if saved is not None and (saved['binding'] != binding or saved['identity'] != ident):
                raise ValueError('Repair checkpoint lineage changed')
            for name in audit.OPTIONS:
                if row['plans'][name]['status'] != 'unresolved':
                    continue
                menu, mode = name.split('_')
                args = (c0, ols[at], bank if menu == 'expanded' else bank[:, :width], y, v, q, *cut)
                bound = row['arbitrary'][mode]
                if saved is not None:
                    a, log = (saved['plans'][name][k] for k in ('actions', 'solver'))
                else:
                    parents = ([f'{menu}_query'] if mode == 'record' else []) + ([f'base_{mode}'] if menu == 'expanded' else [])
                    inherited = [solved[k]['actions'] for k in parents if k in solved and solved[k]['actions']]
                    design = m.design(*args[:-1], edits, radii, mode)
                    candidate, log = kernel(design, nearby, seconds)
                    good = []
                    for a in ([candidate] if candidate else []) + inherited:
                        z = audit.verify(a, args, mode, group, edits, radii, bound)
                        if z['verified']:
                            good.append((min(z['gains']), a))
                    a = max(good, key=lambda g: g[0])[1] if good else None
                receipt = audit.verify(a, args, mode, group, edits, radii, bound) if a else None
                if receipt is not None and not receipt['verified']:
                    raise ValueError('Repair failed original verifier')
                solved[name] = dict(actions=a, solver=log, witness=receipt)
            old.once(cp, old.stable(dict(binding=binding, identity=ident, plans=solved)))
            rows.append(dict(identity=ident, plans={k:{a:b for a,b in z.items() if a != 'actions'} for k,z in solved.items()}))
            print(dict(recording=rec, seed=i['seed'], verified=[k for k,z in solved.items() if z['witness']]), flush=True)
    counts = {k: sum(bool(z['plans'].get(k, {}).get('witness')) for z in rows) for k in audit.OPTIONS}
    if len(rows) != sum(len(z) for z in by_cell.values()):
        raise ValueError('Not all unresolved views checked')
    old.once(audit.OUT / 'repair.json', old.stable(dict(binding=binding, rows=rows, added_verified=counts,
        examined_views=len(rows), no_exclusion_from_kernel=True, original_2pct=True, hindsight_only=True,
        a_labels_opened=False, confirmation_access=False, independent_confirmation=False)))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
