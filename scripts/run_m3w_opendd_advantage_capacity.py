"""Outcome-aware risk capacity at the exact matched query counts."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_candidate_advantage as prior

run, parent, paired, two = prior.run, prior.parent, prior.paired, prior.two
OUT = prior.OUT
CAP = .02


def solve(groups, counts, gain, risk, support, base_ols, easy_mass):
    """Maximize the lesser of the two lower gains under the unmodified risk caps."""
    n = len(groups)
    if not support.any():
        return "no_eligible_easy", None
    g = np.arange(len(counts))[:, None] == groups[None]
    h = np.c_[g, np.zeros(len(g))]
    low = np.r_[counts, [-np.inf]*5, 1.]
    hi = np.r_[counts, 0., 0., CAP*easy_mass, 0., base_ols, np.inf]
    v = np.r_[h, np.c_[risk.T, np.zeros(3)], np.c_[-gain.T, np.ones(2)], np.r_[support, 0.][None]]
    scale = np.maximum(np.max(np.abs(v), axis=1), 1.)
    result = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
                  bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
                  constraints=LinearConstraint(v/scale[:, None], low/scale, hi/scale),
                  options=dict(time_limit=5., mip_rel_gap=0.))
    if result.status == 2:
        return "risk_infeasible", None
    if result.x is not None:
        on = result.x[:-1] > .5
        margin = (gain[on].sum(0)+[0., base_ols]).min()
        if (margin > 0. and support[on].any() and np.array_equal(g @ on.astype(int), counts)
                and (risk[on].sum(0) <= [0., 0., CAP*easy_mass]).all()):
            return "qualified", on
    bound = getattr(result, "mip_dual_bound", None)
    if bound is not None and np.isfinite(bound) and -bound < -1e-8:
        return "not_dual_positive", None
    return "unresolved", None


def count(d, cv, ols, neural, on, cuts):
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    legal = (cv != neural).any((1, 2))
    m = two.moments(cv, neural, y, valid, cuts[0])
    risk = np.column_stack([m[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    alt = [paired(p, neural, y, valid)[1].sum(1) for p in (cv, ols)]
    base = paired(ols, cv, y, valid)[1].sum(1)
    gain = np.column_stack([alt[0], alt[1]-base])
    counts = np.bincount(q[on], minlength=q.max()+1)
    out = Counter()
    for r in np.unique(rec):
        full = np.flatnonzero(rec == r)
        active = full[(counts[q[full]] > 0) & legal[full]]
        if not len(active):
            out["no_edits"] += 1
            continue
        qs, group = np.unique(q[active], return_inverse=True)
        status, take = solve(group, counts[qs], gain[active], risk[active], m["easy"][active],
                             base[full].sum(), m["ref_easy"][full].sum())
        if status == "qualified":
            p = cv[full].copy()
            p[np.searchsorted(full, active[take])] = neural[active[take]]
            exact = parent.score(dict(y=y[full], valid=valid[full], rec=rec[full]), cv[full], ols[full], p, cuts)
            if exact["dual_positive"] != 1 or exact["edits"] != int(counts[qs].sum()):
                raise ValueError("Capacity witness failed the original scoring")
        out[status] += 1
    return dict(out)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, old_hard, old_heads, old, source = prior.register()
    p = OUT / "capacity_registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), parent_registration=key,
        outcome_aware_only=True, train=False, original_2pct=True, confirmation_access=False,
        scope="same raw candidate and exact per-query count as hardness policy; optimize lesser lower gain",
        undefined="no edit or no eligible selected easy reference cannot pass")))
    sites = {s: prior.forecast.load(s, source)[0] for s in parent.source.SITES}
    result = {}
    for held, d in sites.items():
        cv, ols, *_ = parent.prior.analytic(d)
        cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != held]), [.25, .75])
        x = torch.from_numpy(prior.prior.model.features(d["h"]))
        result[held] = []
        for row in a["by_site"][held]:
            seed = row["seed"]
            cell = next(v for v in old["by_site"][held] if v["seed"] == seed)
            with np.load(run.checked(cell["prediction"]), allow_pickle=False) as z:
                ego = z["comoving_ego"]
            h = two.fit.load(old_heads[f"{held}_{seed}"], lambda: prior.prior.model.network(x.shape[1]), prior.prior.STEPS)
            with torch.inference_mode():
                z = h(x).softmax(1).numpy()
            on = prior.prior.model.decide(z, (cv != ego).any((1, 2)))
            ref = next(v for v in old_hard["by_site"][held] if v["seed"] == seed)["raw"]["gated"]
            assert on.sum() == ref["edits"]
            s = count(d, cv, ols, ego, on, cuts)
            if sum(s.values()) != ref["views"] or s.get("qualified", 0)+s.get("unresolved", 0) < ref["dual_positive"]:
                raise ValueError("Exact assignment failed to include the original policy witness")
            result[held].append(dict(seed=seed, counts=s))
        print(held, result[held], flush=True)
    total = sum((Counter(c["counts"]) for v in result.values() for c in v), Counter())
    run.once(OUT / "capacity.json", run.stable(dict(registration=run.sha(p), by_site=result, summary=total,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))


if __name__ == "__main__":
    main()
