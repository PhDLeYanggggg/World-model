"""Hold the learned query cap fixed and separate risk and gain errors."""

from pathlib import Path
import sys

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_reference_bridge_policy as study

run, np, torch = study.run, study.np, study.torch


def exact(p, q, y, valid, rec, cut):
    cv, ols, bank = (p[k] for k in ("cv", "ols", "bank"))
    m = study.capacity.coefficients(cv, ols, bank, y, valid, cut[0])
    row, arm = m["row"], m["arm"]
    n, rows = len(row), len(cv)
    if not m["easy"].any():
        return "absent_easy" if not m["easy_count"] else "no_legal_easy"
    _, qid, size = np.unique(q, return_inverse=True, return_counts=True)
    cap = np.minimum(2, size-1)
    cols = np.arange(n)
    qmat = coo_array((np.ones(n), (qid[row], cols)), shape=(len(cap), n+1))
    amat = coo_array((np.ones(n), (row, cols)), shape=(rows, n+1))
    s = np.abs(m["risk"]).max(1).clip(1.)
    risk = csc_array(np.column_stack((m["risk"]/s[:, None], np.zeros(3))))
    support = csc_array(np.r_[m["easy"], 0.][None])
    obj = csc_array(np.column_stack((-m["gain"]/m["cv_mass"], np.ones(2))))
    mat = vstack((qmat, amat, risk, support, obj), format="csc")
    lower = np.r_[np.full(len(cap)+rows+3, -np.inf), 1., -np.inf, -np.inf]
    upper = np.r_[cap, np.ones(rows), 0., 0., m["easy_cap"]/s[2], np.inf, m["offset"]/m["cv_mass"]]
    sol = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(mat, lower, upper), options=dict(time_limit=2., mip_rel_gap=0.))
    if sol.x is not None:
        at = sol.x[:-1] > .5
        pred = cv.copy()
        pred[row[at]] = bank[row[at], arm[at]]
        edits = (pred != cv).any((1, 2))
        if edits.sum() != at.sum() or np.any(np.bincount(qid[edits], minlength=len(cap)) > cap):
            raise ValueError("Candidate witness violates exact count constraints")
        v = study.capacity.policy(pred, cv, cv, y, valid, np.full(rows, rec), easy_cut=cut[0], hard_cut=cut[1])["recordings"][rec]
        g = 100*np.array([study.transfer.paired(a, pred, y, valid)[1].sum() for a in (cv, ols)])/m["cv_mass"]
        np.testing.assert_allclose(g, 100*(m["offset"]+m["gain"][:, at].sum(1))/m["cv_mass"], atol=1e-10, rtol=1e-8)
        if v["supported"] and g.min() > study.capacity.POSITIVE_PP:
            return "verified_positive_both"
    if sol.status == 2:
        return "risk_infeasible"
    dual = getattr(sol, "mip_dual_bound", None)
    if dual is not None and -100*dual <= study.capacity.POSITIVE_PP:
        return "no_strict_positive_bound"
    return "unresolved"


def score(d, p, cost, cut):
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    cv, ols, bank = (p[k] for k in ("cv", "ols", "bank"))
    labels = study.model.targets(cv, ols, bank, y, valid, cut[0], p["scale"])*p["scale"][:, None, None]
    masks = dict(oracle_risk=np.concatenate((cost[..., :2], labels[..., 2:]), axis=2),
                 oracle_gain=np.concatenate((labels[..., :2], cost[..., 2:]), axis=2))
    out = {}
    for k, m in masks.items():
        c = study.model.query_choices(m, p["legal"], q, p["motion"])["joint"]
        at = c >= 0
        path = cv.copy()
        path[at] = bank[at, c[at]]
        v = study.capacity.policy(path, cv, cv, y, valid, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
        passed = [name for name, a in v.items() if a["supported"]]
        both = sum(all(study.transfer.paired(ref[rec == name], path[rec == name], y[rec == name], valid[rec == name])[1].sum() > 0.
                       for ref in (cv, ols)) for name in passed)
        out[k] = dict(edits=int(at.sum()), pass_views=len(passed), positive_both=both)
    exact_results = {}
    for name in sorted(set(rec)):
        at = rec == name
        k = exact({k: a[at] for k, a in p.items() if k in ("cv", "ols", "bank")}, q[at], y[at], valid[at], name, cut)
        exact_results[k] = exact_results.get(k, 0)+1
    return dict(counterfactual=out, exact_counts=exact_results)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, _ = study.register()
    files = [study.OUT / n for n in ("analysis.json", "head_freeze.json")]
    a, h = [run.read(p) for p in files]
    if a["registration"] != key or a["heads"] != run.sha(files[1]) or a["confirmation_access"]:
        raise ValueError("Require bound held-location policy")
    f = study.OUT / "bottleneck_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=[run.artifact(p) for p in files],
        cap_identical_to_learned=True, counterfactual_uses_future_for_diagnostics_only=True,
        original_2pct=True, confirmation_access=False)))
    nets = study.transfer.frozen_models(*frozen[:5])
    data = {s: study.old.load_site(s) for s in study.source.SITES}
    rows = []
    for i, held in enumerate(study.source.SITES):
        cut = np.quantile(np.concatenate([v["cut_samples"] for k, v in data.items() if k != held]), [.25, .75])
        for seed in study.old.prior.SEEDS:
            d = data[held]
            p = study.prepare(d, study.transfer.forecast(d["h"], [nets[seed][i]]))
            meta = h["heads"][f"{held}/{seed}"]
            z = torch.load(run.checked(meta["checkpoint"]), weights_only=True, map_location="cpu")
            if run.stable(z["identity"]) != meta["identity"] or z["step"] != study.STEPS:
                raise ValueError("Unexpected held checkpoint")
            net = study.model.BridgeValue()
            net.load_state_dict(z["net"])
            with torch.inference_mode():
                cost = net.eval()(torch.from_numpy(p["x"])).numpy()*p["scale"][:, None, None]
            rows.append(dict(site=held, seed=seed, **score(d, p, cost, cut)))
    counts = {k: sum(r["exact_counts"].get(k, 0) for r in rows) for k in sorted(set().union(*(v["exact_counts"] for v in rows)))}
    if sum(counts.values()) != 606:
        raise ValueError("Missing recording-seed views")
    c = {k: {f2: sum(r["counterfactual"][k][f2] for r in rows) for f2 in rows[0]["counterfactual"][k]}
         for k in rows[0]["counterfactual"]}
    run.once(study.OUT / "bottleneck.json", run.stable(dict(registration=run.sha(f), exact=counts, counterfactual=c,
        by_site=rows, no_training=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
