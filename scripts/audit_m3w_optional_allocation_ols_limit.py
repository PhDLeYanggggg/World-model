"""Exact OLS4-only ceilings without risk restrictions at nested query caps."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_optional_allocation as study

run, job, np = study.run, study.job, study.np
CAPS = (2, 4, 8, 16, None)


def max_additive_gain(increment, inverse, counts):
    if len(increment) != len(inverse) or not np.isfinite(increment).all():
        raise ValueError("Aligned finite per-row gains required")
    n = len(increment)
    order = np.lexsort((np.arange(n), -increment, inverse))
    sizes = np.bincount(inverse, minlength=len(counts))
    ranks = np.arange(n)-np.repeat(np.r_[0, np.cumsum(sizes)[:-1]], sizes)
    ids = order[(ranks < counts[inverse[order]]) & (increment[order] > 0.)]
    return float(increment[ids].sum())


def upper_limits(cv, ols, bank, target, valid, inverse, counts, w):
    known = valid.any(1)
    b = job.errors(cv, target, valid)[0]
    o = job.errors(ols, target, valid)[0]
    d_ols = np.linalg.norm(ols-cv, axis=-1)
    offset = float(w@np.where(known, o-b, -d_ols.max(1)))
    norm = np.linalg.norm(bank-cv[:, None], axis=-1)
    legal = norm.max(2) > 0.
    e = np.stack([job.errors(p, target, valid)[0] for p in bank.swapaxes(0, 1)], axis=1)
    flat = np.where(known[:, None], b[:, None]-e,
                    d_ols.max(1)[:, None]-np.linalg.norm(ols[:, None]-bank, axis=-1).max(2))
    d_y = np.linalg.norm(np.where(valid[..., None], target-cv, 0.), axis=-1)
    circle_e = (np.abs(norm-d_y[:, None])*valid[:, None]).sum(2)/valid.sum(1).clip(1)[:, None]
    circle = np.where(known[:, None], b[:, None]-circle_e,
                      d_ols.max(1)[:, None]-np.abs(norm-d_ols[:, None]).max(2))
    den = float(w@np.where(known, b, 0.))
    out = {}
    for name, gain in (("quarter", flat), ("circle", circle)):
        best = np.where(legal, gain, 0.).max(1).clip(0.)*w
        out[name] = [offset+max_additive_gain(best, inverse, counts if k is None else np.minimum(k, counts))
                     for k in CAPS]
    if np.any(np.array(out["circle"])+1e-7 < out["quarter"]):
        raise ValueError("Free directions must retain every quarter-path choice")
    return dict(reference_mass=den, **out)


def main():
    binding, identity, _, _ = study.register()
    rows = []
    for rec, g, _, cv, ols, bank, target, valid, q, counts, w in job.prior.prior.old.blocks(
            identity, lambda **_: None):
        p = cv[:, None]+.25*(bank-cv[:, None])
        rows.append(dict(recording=rec, source=g, **upper_limits(cv, ols, p, target, valid, q, counts, w)))
    if len(rows) != 24:
        raise ValueError("Require the original development population")
    summary = {}
    for name in ("quarter", "circle"):
        values = np.array([r[name] for r in rows])
        tol = 1e-7*np.maximum(np.array([r["reference_mass"] for r in rows]), 1.)[:, None]
        summary[name] = dict(positive=(values > tol).sum(0).tolist(),
                             negative=(values < -tol).sum(0).tolist(),
                             near_zero=(np.abs(values) <= tol).sum(0).tolist())
    run.once(study.OUT / "ols_limit.json", run.stable(dict(registration=binding,
        code=run.artifact(Path(__file__)), query_caps=CAPS, summary=summary, recordings=rows,
        exact_ols_only_objective_without_risk_or_cv=True, label_aware_not_a_policy=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
