"""Compute the exact one-edit ceiling over CV-centered displacement balls."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_sustained_turn as prior
from src.evaluation.m3w_recording_switch_ceiling import moments

run, parent, paired = prior.run, prior.parent, prior.paired
study = prior.prior.study
OUT = prior.OUT / "radius_ceiling_v2"


def reachable(cv, ols, y, valid, cap, easy_cut):
    """Solve the within-row missing-step tradeoff, without changing the risk rule."""
    e = np.linalg.norm(cv-y, axis=-1)
    observed = np.where(valid, e, 0.)
    c = np.minimum(observed, cap[:, None]).sum(1)
    dist = np.linalg.norm(ols-cv, axis=-1)
    movable = np.where(~valid, np.minimum(dist, cap[:, None]), 0.)
    # At a known step the projection towards the label minimizes both errors.
    # At missing steps moving x towards OLS spends x of the CV gain and adds x to OLS gain.
    offset = paired(ols, cv, y, valid)[1].sum(1)
    o = offset.sum()+c
    lim = movable.sum(1)
    easy = moments(cv, cv, y, valid, easy_cut)["easy"]
    possible = easy & (c > 0.) & ((o+np.minimum(c, lim)) > 0.)
    return possible, c, o, movable, observed


def witness(cv, ols, y, valid, cap, c, o, movable, observed):
    """Construct a single forecast on the open positive-gain frontier."""
    gap = min(c, movable.sum())
    x = 0. if o > 0. else .5*(-o+gap)
    taken = np.clip(x-np.r_[0., movable[:-1].cumsum()], 0., movable)
    move = np.minimum(observed, cap)
    known_fraction = np.divide(move, observed, out=np.zeros_like(move), where=observed > 0.)
    d = np.linalg.norm(ols-cv, axis=-1)
    missing_fraction = np.divide(taken, d, out=np.zeros_like(taken), where=d > 0.)
    return cv + np.where(valid[:, None], (y-cv)*known_fraction[:, None],
                         (ols-cv)*missing_fraction[:, None])


def recording(cv, ols, y, valid, cap, cuts):
    bound, c, o, movable, observed = reachable(cv, ols, y, valid, cap, cuts[0])
    if not bound.any():
        return False
    i = np.flatnonzero(bound)[0]
    p = witness(cv[i], ols[i], y[i], valid[i], cap[i], c[i], o[i], movable[i], observed[i])
    delta = np.linalg.norm(p-cv[i], axis=-1).max()
    if delta > cap[i]+1e-10 or not prior.prior.one_edit(cv, ols, y, valid, np.array([i]), p[None], cuts)["both"]:
        raise ValueError("Analytic radius witness failed the original one-edit scorer")
    return True


@torch.inference_mode()
def execute(key, a, ph, h, old, prev, source):
    observed = run.read(prior.OUT / "analysis.json")
    if observed["registration"] != key or observed["confirmation_access"]:
        raise ValueError("Require verified curvature readout")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), source=run.artifact(prior.OUT / "analysis.json"),
        previous_aborted=run.artifact(prior.OUT / "radius_ceiling" / "registration.json"),
        assumptions="at most one eligible edited row and max step offset bounded by largest legal OLS bridge radius",
        label_use="construct hindsight witness for capacity only; no inference output",
        comparisons="bounded and unbounded analytic trajectory space vs finite banks, same masks and 2% rule",
        original_2pct=True, confirmation_access=False, training=False)))
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    table = {}
    for s, d in sites.items():
        bank = parent.prior.analytic(d)
        cv, ols, linear, _, _, legal = bank
        max_r = np.where(legal, prior.model.legacy.peak(linear-cv[:, None]), 0.).max(1)
        cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
        with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
            rec = z["recording"]
        cells = []
        for cell in a["by_site"][s]:
            seed = cell["seed"]
            rank = study.load(ph["heads"][f"{s}/{seed}/additive_rank"],
                              lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            menu = study.menus(d, bank, rank)
            active = menu["legal"][:, 1]
            names, inv = np.unique(rec, return_inverse=True)
            short = names[np.bincount(inv, weights=active) < 5]
            count = Counter(views=len(short), bounded=0, unbounded=0)
            for name in short:
                at = d["rec"] == name
                cap = np.where(active[d["query"][at]], max_r[at], 0.)
                args = cv[at], ols[at], d["y"][at], d["valid"][at]
                finite = recording(*args, cap, cuts)
                loose = recording(*args, np.where(cap > 0., np.inf, 0.), cuts)
                if finite and not loose:
                    raise ValueError("Removing a bound lost an attainable forecast")
                count.update(bounded=int(finite), unbounded=int(loose))
            ref = next(r for r in observed["by_site"][s]["cells"] if r["seed"] == seed)["oracle"]
            if count["views"] != ref["views"] or count["bounded"] < max(ref.get(k, 0) for k in ("anchor", *prior.model.NAMES)):
                raise ValueError("Analytic ceiling contradicts the finite-bank experiment")
            cells.append(dict(seed=seed, **count))
        table[s] = cells
    totals = sum((Counter({k: v for k, v in c.items() if k != "seed"})
                  for values in table.values() for c in values), Counter())
    run.once(OUT / "analysis.json", run.stable(dict(registration=run.sha(f), summary=totals, by_site=table,
        independent_confirmation=False, original_2pct=True, deployment=False)))
    print(dict(totals))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    execute(*prior.register())


if __name__ == "__main__":
    main()
