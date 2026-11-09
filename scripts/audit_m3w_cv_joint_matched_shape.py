"""Generous analytic rematch of confirmed new neural witnesses, post hoc."""

import collections
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_feasibility as study
from src.evaluation.m3w_joint_path_audit import query as pair_query

np, read = study.np, study.read


def pair_masses(history, paths, truth, valid, queries):
    sums = np.zeros((2, 4))
    for q in np.unique(queries):
        ix = queries == q
        _, scores, changed, (steps, known, _, _) = pair_query(history[ix], paths[:, ix], truth[ix], valid[ix], ((0, 2), (1, 2)))
        sums[:, 0] += np.where(changed & (known > 0), np.nan_to_num(scores[:, 0]) * known, 0.)
        sums[:, 1] += np.where(changed & (steps > 0), np.nan_to_num(scores[:, 1]) * steps, 0.)
        sums[:, 2] += changed * known
        sums[:, 3] += changed * steps
    return sums


def rematch(cv, ols, analytic, original, edits, y, valid, query, easy, hard, cap):
    row, arm = np.asarray(edits, dtype=int).T
    selected = original[row, arm]
    peak = np.linalg.norm(selected - cv[row], axis=-1).max(1)
    step = analytic[row] - cv[row, None]
    native = np.linalg.norm(step, axis=-1).max(-1)
    if not (native > 0).any(1).all():
        return 'no_analytic_direction'
    ratio = np.divide(peak[:, None], native, out=np.zeros_like(native), where=native > 0.)
    bank = np.broadcast_to(cv[:, None], analytic.shape).copy()
    bank[row] = cv[row, None] + step * ratio[..., None, None]
    # Keep unchanged native actions bit-identical whenever they originally matched the peak.
    bank[row] = np.where((native == peak[:, None])[..., None, None], analytic[row], bank[row])
    actual_peak = np.linalg.norm(bank[row] - cv[row, None], axis=-1).max(-1)
    np.testing.assert_allclose(actual_peak, np.where(native > 0., peak[:, None], 0.), rtol=1e-8, atol=1e-10)
    d = study.model.coefficients(cv, ols, bank, y, valid, query, easy)
    if not d['easy'].any():
        return 'undefined_support'
    mat, rhs = study.model.constraints(d, cap)
    low = np.full(len(rhs), -np.inf)
    low[row] = 1.
    n = len(d['row'])
    s = study.model.milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=study.model.Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=study.model.LinearConstraint(mat, low, rhs),
        options={'time_limit': study.SECONDS, 'mip_rel_gap': 0.})
    if s.status == 2:
        return 'risk_infeasible'
    if s.x is not None:
        take = s.x[:-1] > .5
        moved = np.column_stack((d['row'][take], d['arm'][take]))
        np.testing.assert_array_equal(np.sort(moved[:, 0]), np.sort(row))
        result = study.verify(moved, cv, ols, bank, y, valid, query, easy, hard, cap)
        if result['verified']:
            return 'verified_analytic_rematch'
    upper = getattr(s, 'mip_dual_bound', None)
    if upper is not None and np.isfinite(upper) and -upper < -study.model.POSITIVE_PP:
        return 'nonpositive_bound'
    return 'unresolved'


def main():
    parent = study.OUT / 'readout.json'
    status = read(parent)
    r = read(study.PRIVATE / 'recording_receipt.json')
    if r['registration'] != status['registration'] or status['independent_confirmation']:
        raise ValueError('Require verified development-only receipt')
    a = study.parent.parent.a_inputs()
    reg = read(study.prior.parent.REG)
    neural = {s: study.prior.parent.read_banks(reg['forecasts'], s, a['global_ids'], a['sites']) for s in study.SEEDS}
    old = study.prior.bank.old
    labels = old.prior.arrays(('target_eval', 'valid', 'origin'))
    truth, valid = labels['target_eval'] - labels['origin'][:, None], labels['valid']
    bank = study.motion.bank(a['history'])
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(v['job']['producer_fold'])] for v in read(old.PARENTS[3])['forecasts']
           for s in v['job']['donor_sites']}
    rec = a['recordings'].astype(str)
    counts = {str(cap): collections.Counter() for cap in study.CAPS}
    pair_sums = {str(cap): np.zeros((2, 4)) for cap in study.CAPS}
    ana = {(v['recording'], v['cap']): v['status'] for v in r['cells'] if v['seed'] is None}
    for v in r['cells']:
        seed, name, cap = (v[k] for k in ('seed', 'recording', 'cap'))
        if seed is None or v['status'] != 'verified' or ana[name, cap] in ('verified', 'unresolved'):
            continue
        stored = read(study.PRIVATE / f'{name}_{seed}_{cap}.json')
        if stored['identity']['registration'] != r['registration'] or stored['witness'] != v['witness']:
            raise ValueError('Witness identity differs')
        ix = rec == name
        local = bank[ix]
        full = np.concatenate((local[:, 1:], np.stack([neural[seed][k][ix] for k in ('eqmotion', 'pct')], axis=1)), axis=1)
        c = cut[v['source']]
        k = rematch(local[:, 0], local[:, 1], local[:, 1:], full, stored['actions'], truth[ix], valid[ix],
                    a['queries'][ix], c['easy'], c['hard'], cap)
        counts[str(cap)][k] += 1
        path = local[:, 0].copy()
        row, arm = np.asarray(stored['actions'], int).T
        path[row] = full[row, arm]
        pair_sums[str(cap)] += pair_masses(a['history'][ix], np.stack((local[:, 0], local[:, 1], path)),
                                           truth[ix], valid[ix], a['queries'][ix])
    if any(sum(c.values()) != status['summary']['caps'][k]['confirmed_new_views'] for k, c in counts.items()):
        raise ValueError('Confirmed new witness roster changed')
    pair = {cap: {name: dict(observed_gain_pixel=x[0]/x[2] if x[2] else None,
                           full_grid_lower_gain_pixel=x[1]/x[3] if x[3] else None,
                           known_pair_steps=int(x[2]), total_pair_steps=int(x[3]))
                  for name, x in zip(('cv', 'ols4'), v, strict=True)} for cap, v in pair_sums.items()}
    study.once(study.OUT / 'matched_shape_diagnostic.json', study.stable(dict(parent=study.artifact(parent),
        code=study.artifact(Path(__file__)), post_hoc=True, recipient_and_each_peak_matched=True,
        analytic_amplitude_may_exceed_native=True, statuses=counts, pair_geometry=pair, independent_confirmation=False)))
    print(counts, pair)


if __name__ == '__main__':
    main()
