"""Bound outcome-aware recording feasibility for the trained controller's menus."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy import sparse
from scipy.optimize import linprog

from scripts import train_m3w_cv_joint_count as study
from src.evaluation.m3w_selection_forecast_readout import errors

old, base, prior, np = study.old, study.base, study.prior, study.np
MODES = ('fixed_count', 'variable_count')


def record_problem(d, cv, ols, bank, packet, cut):
    all_terms, all_plans, roster = [], [], []
    offset, ref, cap = 0., 0., 0.
    for p in packet:
        at = p['at']
        c, o, y, v = cv[at], ols[at], d['target'][at], d['valid'][at]
        forecast = study.paths(c, bank[at], p['plans'])
        risk = study.risk_excess(c, forecast.swapaxes(0, 1), y, v, cut)
        m, off, g, _ = study.candidate_terms(c, o, forecast.swapaxes(0, 1), y, v)
        cv_err, _ = errors(c, y, v)
        easy = v.any(1) & (cv_err > 0.) & (cv_err <= cut)
        observed = np.where(v[..., None], y, 0.)
        err = (np.linalg.norm(forecast - observed, axis=-1) * v).sum(2) / v.sum(1).clip(1)
        signed = np.where(easy, err - cv_err, 0.).sum(1)
        support = ((p['plans'] >= 0) & easy).sum(1)
        all_terms.append(np.column_stack((g.mean(0), risk.sum(0), signed, support)))
        all_plans.append(p['plans'])
        roster.append(at)
        offset += off.mean()
        ref += m.mean()
        cap += .02 * cv_err[easy].sum()
    return all_terms, all_plans, roster, ref, offset, cap


def search(blocks, ref, offset, cap):
    if ref <= 0. or sum(x[:, 5].max() for x in blocks) < 1.:
        return dict(status='unsupported', upper=None), None
    v = np.concatenate(blocks)
    g = v[:, :2] * (100. / ref)
    offset *= 100. / ref
    divisor = max(ref / 12., cap, 1.)
    c = np.column_stack((v[:, 2:5] / divisor, -v[:, 5]))
    budget = np.array([0., 0., cap / divisor, -1.])
    limits = np.array([0., offset, *budget])
    constraints = np.column_stack((np.vstack((-g.T, c.T)), [1., 1., 0., 0., 0., 0.]))
    counts = np.array([len(z) for z in blocks])
    start = np.r_[0, counts.cumsum()]
    columns = np.arange(len(v))
    owners = np.repeat(np.arange(len(counts)), counts)
    exact_one = sparse.coo_matrix((np.ones(len(v)), (owners, columns)), shape=(len(counts), len(v)+1)).tocsr()
    objective = np.r_[np.zeros(len(v)), -1.]
    solution = linprog(objective, A_ub=constraints, b_ub=limits, A_eq=exact_one,
        b_eq=np.ones(len(counts)), bounds=[(0., 1.)] * len(v) + [(None, None)], method='highs')
    unconstrained = np.minimum(np.maximum.reduceat(g[:, 0], start[:-1]).sum(),
        offset + np.maximum.reduceat(g[:, 1], start[:-1]).sum())
    if not solution.success:
        return dict(status='lp_infeasible' if solution.status == 2 else 'unresolved',
                    upper=None, unconstrained=float(unconstrained)), None
    # Any nonnegative dual mixture yields an upper bound, even if the LP
    # solution is imperfect. Recompute the bound from actual finite menus.
    dual = np.maximum(-solution.ineqlin.marginals, 0.)
    if dual[:2].sum() <= 0.:
        raise ValueError('Invalid dual gain mixture')
    dual /= dual[:2].sum()
    upper = np.maximum.reduceat(g @ dual[:2] - c @ dual[2:], start[:-1]).sum()
    upper += limits @ dual
    if upper + 1e-6 < -solution.fun:
        raise ValueError('Dual certificate is below primal score')
    rounded = [int(solution.x[a:b].argmax()) for a, b in zip(start[:-1], start[1:], strict=True)]
    return dict(status='excluded' if upper <= 1e-7 else 'open', upper=float(upper),
                unconstrained=float(unconstrained)), rounded if upper > 1e-7 else None


def check(d, cv, ols, bank, roster, plans, rounded, cut):
    scorer = prior.prior.parent.readout
    at = np.sort(np.concatenate(roster))
    actions = np.full(len(cv), -1, np.int16)
    for ix, p, k in zip(roster, plans, rounded, strict=True):
        actions[ix] = p[k]
    y, v, q = (d[k][at] for k in ('target', 'valid', 'query'))
    z = study.parent.model.apply(cv[at], bank[at], actions[at])
    r = scorer.score_recording({base.CV: cv[at], 'p': z}, {}, y, v, q, str(d['recording'][at[0]]),
        easy_cut=cut[0], hard_cut=cut[1])
    if not r['metrics']['p']['risk']['supported']:
        return None
    gain = [scorer.merge_comparisons([prior.study.motion.full_masses(left[at], z, cv[at], y, v, old.weight(q))])
            ['completion_lower_pp'] for left in (cv, ols)]
    return gain if min(gain) > 1e-7 else None


def main():
    study.torch.set_num_threads(4)
    f, r = (study.OUT / k for k in ('freeze.json', 'readout.json'))
    freeze, result = map(old.read, (f, r))
    if (result['freeze'] != old.sha(f) or not freeze['before_held_scoring'] or not result['original_2pct']
            or result['a_labels_opened'] or result['independent_confirmation']):
        raise ValueError('Require frozen held development results')
    reg = old.read(study.OUT / 'registration.json')
    parent = old.read(old.checked(reg['parents'][1]))
    base.register()
    d = base.fit_population(lambda **_: None)
    out = {k: [] for k in MODES}
    for cell, anchor in zip(freeze['cells'], parent['cells'], strict=True):
        ident, cut = cell['identity'], cell['cut']
        if anchor['identity'] != ident:
            raise ValueError('Shape fit identity mismatch')
        _, h = base.subset(d, ident['held'])
        x, _ = study.parent.parent.context.context(h)
        shape = study.loader.load(anchor['fits'][1], x.shape[1] + 37)
        cv, ols, bank, _, packet = study.packet(h, shape)
        if sum(len(p['plans']) for p in packet) != cell['held_options']:
            raise ValueError('Frozen slate menu size changed')
        for rec in np.unique(h['recording']):
            p = [x for x in packet if h['recording'][x['at'][0]] == rec]
            blocks, plans, roster, ref, offset, cap = record_problem(h, cv, ols, bank, p, cut[0])
            for mode in MODES:
                idx = [x['counts'] == x['anchor_count'] for x in p] if mode == 'fixed_count' else [
                    np.ones(len(x['counts']), bool) for x in p]
                b, z = [[x[t] for x, t in zip(values, idx, strict=True)] for values in (blocks, plans)]
                row, rounded = search(b, ref, offset, cap)
                if rounded is not None:
                    witness = check(h, cv, ols, bank, roster, z, rounded, cut)
                    if witness is not None:
                        row.update(status='verified', witness_pp=witness)
                out[mode].append(row)
        print(dict(fold=ident['fold'], seed=ident['seed'], views=len(out['fixed_count'])), flush=True)
    if len(out['fixed_count']) != 72:
        raise ValueError('Incomplete capacity roster')
    summary = {}
    for k, rows in out.items():
        s = {a: sum(r['status'] == a for r in rows) for a in
            ('verified', 'excluded', 'unsupported', 'lp_infeasible', 'open', 'unresolved')}
        s['unconstrained_nonpositive'] = sum(r.get('unconstrained', np.inf) <= 0. for r in rows)
        summary[k] = s
    old.once(study.OUT / 'capacity.json', old.stable(dict(parent=old.artifact(r), code=old.artifact(Path(__file__)),
        views=72, summary=summary, labels_hindsight_only=True, independent_confirmation=False,
        a_labels_opened=False, original_2pct=True)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
