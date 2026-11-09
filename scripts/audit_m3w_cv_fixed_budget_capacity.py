"""Outcome-aware fractional upper bound, with exactly the frozen query constraints."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_fixed_budget_gain as m

np, old = m.np, m.old


def main():
    m.torch.set_num_threads(4)
    p = m.OUT / 'readout.json'
    f = m.OUT / 'freeze.json'
    z, freeze = old.read(p), old.read(f)
    if (z['freeze'] != old.sha(f) or not z['hindsight_diagnostic_only']
            or z['a_labels_opened'] or z['independent_confirmation']):
        raise ValueError('Require frozen development readout')
    rp, _, _, _, sp = [old.checked(k) for k in old.read(m.OUT / 'registration.json')['parents']]
    m.base.register()
    data = m.base.fit_population(lambda **_: None)
    totals, queries, positive = {}, 0, 0
    for c, rc, sc in zip(freeze['cells'], old.read(rp)['cells'], old.read(sp)['cells'], strict=True):
        ident = c['identity']
        _, h = m.base.subset(data, ident['held'])
        cv, ols, b, pred, groups, anchor = m.packet(h, *m.parents(rc, sc, h))
        with np.load(old.checked(c['choices']), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['ids'], h['ids'])
            np.testing.assert_array_equal(saved['anchor'], anchor)
        gain, risk, legal, mass = pred
        true = m.scores(h, cv, ols, b, legal, {}, hindsight=True)['hindsight']
        upp = np.zeros(len(h['ids']))
        for q in groups:
            if not (anchor[q] >= 0).any():
                continue
            budget = m.model.Budget(legal[q], risk[q], mass[q], gain[q, :, 0], anchor[q])
            row = budget.row[budget.edit]
            score = np.zeros(len(budget.row))
            score[budget.edit] = true[q[row], budget.action[budget.edit]]
            val = m.model.linprog(-score, A_ub=budget.col.T, b_ub=budget.limit,
                A_eq=budget.equal, b_eq=np.r_[np.ones(budget.n), budget.count],
                bounds=(0., 1.), method='highs')
            if not val.success:
                raise RuntimeError(val.message)
            inc = -val.fun-score[budget.anchor].sum()
            if inc < -1e-6:
                raise ValueError('Upper bound below the feasible incumbent')
            upp[q[0]] = 12*max(0., inc)/len(q)
            queries += 1
            positive += int(inc > 1e-6)
        w = old.weight(h['query'])
        outcome = h['valid']
        target = np.where(outcome[..., None], h['target'], 0.)
        denominator = w*np.where(outcome, np.linalg.norm(cv-target, axis=-1), 0.).sum(1)
        for site in np.unique(h['sites']):
            at = h['sites'] == site
            if denominator[at].sum() <= 0.:
                raise ValueError('Undefined CV reference')
            totals[str(site), ident['seed']] = float(100*upp[at].sum()/denominator[at].sum())
        print(dict(fold=ident['fold'], seed=ident['seed'], active_queries=queries), flush=True)
    sites = sorted({s for s, _ in totals})
    values = np.array([[totals[s, k] for k in m.base.SEEDS] for s in sites])
    result = dict(parent=old.artifact(p), code=old.artifact(Path(__file__)), active_queries=queries,
        positive_relaxation_queries=positive, mean_upper_increment_ols4_pp=float(values.mean()),
        max_source_seed_upper_increment_ols4_pp=float(values.max()),
        increment_ols4_pp_by_source_seed={s: row.tolist() for s, row in zip(sites, values, strict=True)},
        objective='conservative score against OLS4, incremental over incumbent, CV mass denominator',
        fractional_relaxation=True, diagnostic_only=True, independent_confirmation=False,
        a_labels_opened=False, original_2pct=True)
    old.once(m.OUT / 'capacity_readout.json', old.stable(result))
    print(result, flush=True)


if __name__ == '__main__':
    main()
