"""Measure the fixed-count analytic-menu opportunity over OLS4."""

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_ols_relative_gain as study
from src.evaluation.m3w_selection_forecast_readout import errors


np, run, cal, target = study.np, study.run, study.cal, study.target
MATCHED = ("cv_gain", "ols_joint")


def exact_count_upper(options, counts):
    """Every selected actor must change path; dropping a forced loss is not allowed."""
    rank = np.sort(options.max(1))[::-1]
    if int(counts.max(initial=0)) > np.isfinite(rank).sum():
        raise ValueError("Infeasible recorded count")
    return np.r_[0., rank[np.isfinite(rank)].cumsum()][counts]


def main():
    key, manifest, old, roster, _ = study.register()
    p = study.PUBLIC/"analysis.json"
    score = run.read(p)
    freeze = run.read(run.checked(score["freeze"]))
    if (score["registration"] != key or freeze["registration"] != key or freeze["labels_used"]
            or score["confirmation_access"] or freeze["confirmation_access"]):
        raise ValueError("Use only the scored and frozen calibration roster")
    meta, provenance, cut = target.context(roster)
    ix = [study.ARMS.index(a) for a in MATCHED]
    accum = defaultdict(list)
    edits = np.zeros(len(MATCHED), dtype=int)
    observed = dict(queries=0, ols4_query_risk_pass=0, relaxed_either_violation=0)
    for i, (cell, f) in enumerate(zip(old["clips"], freeze["records"], strict=True), 1):
        rec = cell["sequence"]
        if rec != f["recording"]:
            raise ValueError("Misaligned recording")
        src = cal.reader(manifest, rec, meta, provenance)
        d = target.old_arrays(cell)
        y, v = cal.label_arrays(src, d)
        c, o, at = (d[k] for k in ("cv", "ols", "offsets"))
        coef = np.load(run.checked(f["coefficients"]), mmap_mode="r", allow_pickle=False)
        x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
        bank = study.model.floor.paths(coef[:, :3])
        ada, _ = errors(o, y, v)
        cv, _ = errors(c, y, v)
        known = v.any(1)
        easy = known & (cv > 0.) & (cv <= cut["easy"])
        gains, excess, mass = [], [], []
        for path in (o, *bank.swapaxes(0, 1)):
            e, _ = errors(path, y, v)
            radius_ols = np.linalg.norm(path-o, axis=-1).max(1)
            gains.append(np.where(radius_ols > 0., np.where(known, ada-e, -radius_ols), -np.inf))
            radius_cv = np.linalg.norm(path-c, axis=-1).max(1)
            changed = radius_cv > 0.
            ref = np.column_stack((np.where(changed & known, cv, 0.), np.where(changed & easy, cv, 0.)))
            h = np.where(known, np.maximum(e-cv, 0.), 0.)
            unknown = np.where(known, 0., radius_cv)
            excess.append(np.column_stack((h+unknown, np.where(easy, h, 0.)+unknown))-.02*ref)
            mass.append(ref)
        menu = np.column_stack(gains[1:])
        m0 = np.add.reduceat(mass[0], at[:-1], axis=0)
        q0 = np.add.reduceat(excess[0], at[:-1], axis=0)
        qmin = np.add.reduceat(np.minimum.reduce(excess), at[:-1], axis=0)
        observed["queries"] += len(q0)
        observed["ols4_query_risk_pass"] += int(((m0 > 0.).all(1) & (q0 <= 0.).all(1)).sum())
        observed["relaxed_either_violation"] += int((qmin > 0.).any(1).sum())
        caps = np.add.reduceat((x[:, :, ix] != 0).astype(int), at[:-1])
        upper = np.zeros((len(study.SEEDS), len(MATCHED)))
        for lo, hi, cap in zip(at[:-1], at[1:], caps, strict=True):
            upper += exact_count_upper(menu[lo:hi], cap)/(hi-lo)
        edits += caps.sum((0, 1))
        w = np.repeat(1./np.diff(at), np.diff(at))
        accum[roster[rec]].append((float(w[known] @ cv[known]), upper.mean(0)))
        if i % 10 == 0:
            print(f"{i}/{len(roster)} candidate-menu bounds", flush=True)
    groups = {g: 100.*sum(x[1] for x in z)/sum(x[0] for x in z) for g, z in sorted(accum.items())}
    rows = np.stack(list(groups.values()))
    boot = np.random.default_rng(target.SEED).integers(len(rows), size=(target.DRAWS, len(rows)))
    summary = {}
    for j, a in enumerate(MATCHED):
        x = rows[:, j]
        if edits[j] != score["metrics"][a]["edits_over_ols"]:
            raise ValueError("Wrong replacement count")
        for g, v in groups.items():
            if v[j]+1e-8 < score["groups"][g][a]["ols4"]:
                raise ValueError("Frozen forecasts exceed their menu ceiling")
        summary[a] = dict(edits_over_ols=int(edits[j]), upper_pp=float(x.mean()),
                          ci95=np.quantile(x[boot].mean(1), [.025, .975]).tolist(), positive_groups=int((x > 0).sum()))
    out = dict(code=run.artifact(Path(__file__)), source=run.artifact(p), arms=MATCHED,
        metrics=summary, observed_risk_query_diagnostic=observed,
        groups={g: x.tolist() for g, x in groups.items()}, result_source="fresh_run",
        interpretation="Exact available-menu maximum at frozen query counts, including missing-label debits and ignoring risk",
        hindsight_only=True, confirmation_access=False)
    run.once(study.PUBLIC/"ceiling.json", run.stable(out))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
