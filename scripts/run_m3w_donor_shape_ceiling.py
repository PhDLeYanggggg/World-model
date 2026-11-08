"""Find the hindsight limit of the same two-shape, fixed-slot policy."""

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
from scripts import run_m3w_donor_shape_choice as prior

run, parent, paired, two, ledger = prior.run, prior.parent, prior.paired, prior.two, prior.ledger
OUT = prior.OUT


def best(gain, risk, base, limit, safety=True):
    """Maximize the lesser of the two conservative gains, with optional risk caps."""
    n = len(gain)
    if n <= 12:
        bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
        feasible = (bits @ risk + base[2:] <= limit).all(1) if safety else np.ones(len(bits), bool)
        if not feasible.any():
            return "infeasible", None
        utility = (bits @ gain + base[:2]).min(1)
        utility[~feasible] = -np.inf
        i = utility.argmax()
        return ("qualified", bits[i].astype(bool)) if utility[i] > 0. else ("no_positive", None)

    # The continuous variable is the common positive margin, not a trajectory choice.
    mat = np.c_[-gain.T, np.ones(2)]
    upper = base[:2]
    if safety:
        mat = np.r_[mat, np.c_[risk.T, np.zeros(len(limit))]]
        upper = np.r_[upper, limit-base[2:]]
    scale = np.maximum(np.maximum(abs(mat).max(1), abs(upper)), 1e-8)
    result = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
                  bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
                  constraints=LinearConstraint(mat/scale[:, None], -np.inf, upper/scale),
                  options=dict(time_limit=5., mip_rel_gap=0.))
    if result.status == 2:
        return "infeasible", None
    if result.status != 0:
        return "unresolved", None
    take = result.x[:-1] > .5
    value = (gain[take].sum(0)+base[:2]).min()
    if safety and not (risk[take].sum(0)+base[2:] <= limit).all():
        return "unresolved", None
    if abs(value) < 1e-9:
        return "unresolved", None
    return ("qualified", take) if value > 0. else ("no_positive", None)


def count(d, bank, packet, menu, counts, cuts):
    cv, ols = bank[:2]
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    use = np.arange(2) < counts[:, None]
    row, arm = menu["row"][use], menu["arm"][use]
    a, b = packet["paths"][row, arm, :2].swapaxes(0, 1)
    m, n = [two.moments(cv[row], p, y[row], valid[row], cuts[0]) for p in (a, b)]
    risk = lambda x: np.column_stack([x[k] for k in prior.RISK])
    base_risk, diff_risk = risk(m), risk(n)-risk(m)
    anchor = cv.copy()
    anchor[row] = a
    gain = np.column_stack([paired(p, anchor, y, valid)[1].sum(1) for p in (cv, ols)])
    gain_b = np.column_stack([paired(p[row], b, y[row], valid[row])[1].sum(1) for p in (cv, ols)])
    delta = gain_b-gain[row]
    easy = two.moments(cv, cv, y, valid, cuts[0])["ref_easy"]
    c = Counter()
    for r in np.unique(rec):
        sub, sel = rec == r, rec[row] == r
        if not easy[row[sel]].sum() > 0.:
            c["no_selected_easy_support"] += 1
            c["zero_edits_without_support" if not sel.any() else "edited_without_support"] += 1
            continue
        base = np.r_[gain[sub].sum(0), base_risk[sel].sum(0)]
        limit = np.array([0., 0., .02*easy[sub].sum()])
        status, chosen = best(delta[sel], diff_risk[sel], base, limit)
        if status == "qualified":
            p = anchor.copy()
            p[row[sel][chosen]] = b[sel][chosen]
            metrics = parent.score(dict(y=y[sub], valid=valid[sub], rec=rec[sub]),
                                   cv[sub], ols[sub], p[sub], cuts)
            if metrics["dual_positive"] != 1 or metrics["edits"] != int(sel.sum()):
                raise ValueError("Hindsight witness failed original exact scorer")
            c["qualified"] += 1
        elif status == "unresolved":
            c["unresolved"] += 1
        else:
            potential, _ = best(delta[sel], diff_risk[sel], base, limit, safety=False)
            key = {"qualified": "risk_prevents_win", "no_positive": "no_dual_utility",
                   "unresolved": "unresolved"}[potential]
            c[key] += 1
    if sum(c[k] for k in ("qualified", "no_selected_easy_support", "risk_prevents_win",
                           "no_dual_utility", "unresolved")) != len(np.unique(rec)):
        raise ValueError("Recording partition does not exhaust the original roster")
    return c


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, ph, h, old, source = prior.register()
    p, reg = [OUT / s for s in ("analysis.json", "ceiling_registration.json")]
    if run.read(p)["registration"] != key:
        raise ValueError("Shape choice registration changed")
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), choice=run.artifact(p),
        scope="original vs donor shape only; same realized edits, same risk and support",
        enumeration="binary enumeration through 12 slots, maximin MILP above; verify every witness",
        no_training=True, confirmation_access=False, original_2pct=True)))
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    output = {}
    for s, d in sites.items():
        use = tuple(k for k in sites if k != s)
        cut = np.quantile(np.concatenate([sites[k]["cut_samples"] for k in use]), [.25, .75])
        bank = parent.prior.analytic(d)
        packet = prior.prepare(sites, s, use, bank)
        with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        output[s] = []
        for x in a["by_site"][s]:
            seed = x["seed"]
            m, n = ledger.frozen_count(d, bank, rec, starts,
                ph["heads"][f"{s}/{seed}/additive_rank"], h["heads"][f"{s}_{seed}_motion"])
            cell = count(d, bank, packet, m, n, cut)
            ref = next(v for v in old["by_site"][s]["cells"] if v["seed"] == seed)
            if cell["no_selected_easy_support"] != ref["policies"]["anchor"]["undefined"]:
                raise ValueError("Shape-invariant support does not match original scorer")
            output[s].append(dict(seed=seed, counts=cell))
        print(s, output[s], flush=True)
    total = sum((r["counts"] for cells in output.values() for r in cells), Counter())
    run.once(OUT / "ceiling.json", run.stable(dict(registration=run.sha(reg), by_site=output, summary=total,
        original_2pct=True, confirmation_access=False, no_training=True)))


if __name__ == "__main__":
    main()
