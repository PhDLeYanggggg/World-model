"""Verify the label-aware capacity at the exact frozen query counts."""

from collections import Counter
from pathlib import Path
import sys

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_matched_bridge_swaps as study

parent, run, np, torch = study.parent, study.run, study.np, study.torch
capacity = parent.prior.prior.capacity


def exact(cv, ols, bank, y, valid, query, count, rec, cuts):
    if not count.any():
        return "zero_switches"
    m = capacity.coefficients(cv, ols, bank, y, valid, cuts[0])
    if not m["easy_count"]:
        return "absent_easy"
    _, q = np.unique(query, return_inverse=True)
    on = count[q[m["row"]]] > 0
    row, arm, r, g, easy = m["row"][on], m["arm"][on], m["risk"][:, on], m["gain"][:, on], m["easy"][on]
    if not easy.any():
        return "no_easy_at_fixed_counts"
    n, rows = len(row), len(cv)
    cols = np.arange(n)
    by_q = coo_array((np.ones(n), (q[row], cols)), shape=(len(count), n+1))
    by_agent = coo_array((np.ones(n), (row, cols)), shape=(rows, n+1))
    s = np.abs(r).max(1).clip(1.)
    risk = csc_array(np.column_stack((r/s[:, None], np.zeros(3))))
    support = csc_array(np.r_[easy, 0.][None])
    utility = csc_array(np.column_stack((-g/m["cv_mass"], np.ones(2))))
    mat = vstack((by_q, by_agent, risk, support, utility), format="csc")
    lower = np.r_[count, np.full(rows+3, -np.inf), 1., -np.inf, -np.inf]
    upper = np.r_[count, np.ones(rows), 0., 0., m["easy_cap"]/s[2], np.inf, m["offset"]/m["cv_mass"]]
    sol = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(mat, lower, upper), options=dict(time_limit=2., mip_rel_gap=0.))
    if sol.x is not None:
        at = sol.x[:-1] > .5
        pred = cv.copy()
        pred[row[at]] = bank[row[at], arm[at]]
        edits = (pred != cv).any((1, 2))
        np.testing.assert_array_equal(np.bincount(q[edits], minlength=len(count)), count)
        if int(edits.sum()) != int(at.sum()):
            raise ValueError("Two active arms for one agent")
        rr = capacity.policy(pred, cv, cv, y, valid, np.full(rows, rec), easy_cut=cuts[0], hard_cut=cuts[1])["recordings"][rec]
        gain = 100*np.array([parent.transfer.paired(p, pred, y, valid)[1].sum() for p in (cv, ols)])/m["cv_mass"]
        np.testing.assert_allclose(gain, 100*(m["offset"]+g[:, at].sum(1))/m["cv_mass"], atol=1e-10, rtol=1e-8)
        if rr["supported"] and gain.min() > capacity.POSITIVE_PP:
            return "verified_positive_both"
    if sol.status == 2:
        return "risk_infeasible"
    dual = getattr(sol, "mip_dual_bound", None)
    if dual is not None and -100*dual <= capacity.POSITIVE_PP:
        return "no_strict_positive_bound"
    return "unresolved"


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, h = study.register()
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    rows, total = {}, Counter()
    for held, d in sites.items():
        cuts = np.quantile(np.concatenate([v["cut_samples"] for s, v in sites.items() if s != held]), [.25, .75])
        bank = parent.prior.analytic(d)
        cv, ols, b, _, _, legal = bank
        q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
        rows[held] = {}
        for cell in frozen["by_site"][held]:
            seed = cell["seed"]
            head = parent.model.Hurdle()
            meta = h["heads"][f"{held}/{seed}/full"]
            z = torch.load(run.checked(meta["artifact"]), weights_only=True, map_location="cpu")
            if z["identity"] != meta["identity"] or z["step"] != parent.STEPS:
                raise ValueError("Unexpected frozen count model")
            head.load_state_dict(z["net"])
            chosen = parent.model.bank_decode(parent.predict(head.eval(), bank), legal, q, [cell["criteria"][1]])[0]
            if int((chosen >= 0).sum()) != cell["scores"]["support_hurdle"]["edits"]:
                raise ValueError("Frozen count disagrees with parent")
            local = Counter()
            for name in sorted(set(rec)):
                ix = rec == name
                _, qq = np.unique(q[ix], return_inverse=True)
                count = np.bincount(qq[chosen[ix] >= 0], minlength=qq.max()+1)
                local[exact(cv[ix], ols[ix], b[ix], y[ix], valid[ix], q[ix], count, name, cuts)] += 1
            rows[held][seed] = dict(local)
            total.update(local)
        print(held, dict(total), flush=True)
    if sum(total.values()) != frozen["summary"]["arms"]["support_hurdle"]["views"]:
        raise ValueError("Missing recording-seed evaluations")
    run.once(study.OUT / "exact_matched_capacity.json", run.stable(dict(registration=key,
        code=run.artifact(Path(__file__)), counts=dict(total), by_site=rows,
        exact_query_count=True, labels_hindsight_only=True, original_2pct=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
