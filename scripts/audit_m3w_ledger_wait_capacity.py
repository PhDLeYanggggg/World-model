"""Measure timing opportunities left by the actual positive-prefix wait decoder."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_ledger_wait as study
from src.evaluation.m3w_selection_forecast_readout import policy

run, torch, parent, timing = study.run, study.torch, study.parent, study.timing
OUT = study.OUT / "capacity"


def one_toggle(cv, ols, y, valid, full_query, paths, args, cuts):
    q, rows, slots, proxy, release, old, eligible, cap = args
    rec = np.zeros(len(cv))

    def readout(p):
        v = policy(p, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]["0.0"]
        return bool(v["supported"] and all(study.prior.paired(ref, p, y, valid)[1].sum() > 0. for ref in (cv, ols)))

    ref = cv.copy()
    ref[rows[old]] = paths[old]
    anchor = readout(ref)
    fixed, optional = anchor, anchor
    tallies = Counter(anchor_joint=int(anchor))
    for i in range(len(old)):
        edges, _, _ = study.model.walk(*args, lambda a, _: a == i)
        if not np.isin(np.delete(old, i), edges).all():
            raise ValueError("Single toggle modified another original edit")
        edited = len(edges) == len(old)
        tallies["moved" if edited else "expired"] += 1
        pred = cv.copy()
        pred[rows[edges]] = paths[edges]
        direct = study.prior.paired(ref, pred, y, valid)[1].sum()
        joint = readout(pred)
        optional |= joint
        fixed |= joint and edited
        tallies["positive_direct_if_moved" if edited else "positive_direct_if_expired"] += int(direct > 0.)
        tallies["improves_anchor_and_joint_matched"] += int(direct > 0. and joint and edited)
    tallies.update(single_toggle_joint_matched=int(fixed), single_toggle_joint_optional=int(optional))
    forward = (proxy > 0.) | np.isin(np.arange(len(rows)), old)
    ceiling = timing.capacity.assess(cv, ols, y, valid, full_query,
                                     eligible, cap, rows[forward], slots[forward], len(old), paths[forward], cuts)
    return tallies, ceiling


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, ph, h, *_ = study.register()
    path = study.OUT / "analysis.json"
    source = run.read(path)
    if source["registration"] != key or source["confirmation_access"]:
        raise ValueError("Require the frozen ledger experiment")
    OUT.mkdir(parents=True, exist_ok=True)
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), source=run.artifact(path),
        primary="label-aware support and utility under positive-prefix destinations and actual decoder",
        fixed_count="same offered slots and radii; single-toggle exact count, and optional expiry as separate diagnostic",
        no_training=True, original_2pct=True, confirmation_access=False)))
    key = run.sha(reg)
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    cells = {}
    for s, d in sites.items():
        b = parent.prior.analytic(d)
        c = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
        f = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(f, allow_pickle=False) as z:
            rec, start = z["recording"], z["start"]
        cells[s] = []
        for cell in a["by_site"][s]:
            seed = cell["seed"]
            m, count = study.frozen_count(d, b, rec, start,
                ph["heads"][f"{s}/{seed}/additive_rank"], h["heads"][f"{s}_{seed}_motion"])
            sums, status = Counter(), Counter()
            for at, _, _, paths, args in study.recordings(d, b, m, count, rec):
                q = d["query"][at]
                value, state = one_toggle(b[0][at], b[1][at], d["y"][at], d["valid"][at], q-q[0], paths, args, c)
                sums.update(value)
                status[state] += 1
            status["zero_edits"] = len(np.unique(rec))-sum(status.values())
            old = next(r for r in source["by_site"][s] if r["seed"] == seed)
            if sums["anchor_joint"] != old["metrics"]["anchor"]["dual_positive"]:
                raise ValueError("Hindsight readout changed the anchor")
            cells[s].append(dict(seed=seed, exact_token_ceiling=status, actual_decoder=sums))
        print(s, flush=True)
    flat = [r for v in cells.values() for r in v]
    summary = {k: sum((Counter(r[k]) for r in flat), Counter()) for k in ("exact_token_ceiling", "actual_decoder")}
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, by_site=cells, summary=summary,
        no_training=True, original_2pct=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
