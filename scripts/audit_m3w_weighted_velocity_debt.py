"""Decompose dual-reference conservative gain into observed gain and path debit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_weighted_velocity as study

job, run = study.job, study.run


def decompose(cv, ols, gap):
    """All inputs share the same CV error denominator and source aggregation."""
    k_cv, l_cv = (cv[k]["point"] for k in ("known_pp", "completion_lower_pp"))
    k_ols, l_ols = (ols[k]["point"] for k in ("known_pp", "completion_lower_pp"))
    total = k_cv+k_ols-l_cv-l_ols
    excess = total-gap
    if excess < -1e-7:
        raise ValueError("Pairwise triangle inequality failed")
    return dict(known_cv=k_cv, known_ols=k_ols, lower_cv=l_cv, lower_ols=l_ols,
        debit_cv=k_cv-l_cv, debit_ols=k_ols-l_ols, unavoidable_sum_debit=gap,
        excess_sum_debit=excess, joint_lower_sum=l_cv+l_ols,
        triangle_optimistic_joint_sum=k_cv+k_ols-gap)


def main():
    key, _, _, _ = study.register()
    p = study.OUT / "analysis.json"
    doc = run.read(p)
    if doc["registration"] != key or doc["confirmation_access"]:
        raise ValueError("Require the matched development readout")
    a = doc["summary"]["effects"]
    pair = a[f"{job.OLS}_vs_cv"]
    gap = pair["known_pp"]["point"]-pair["completion_lower_pp"]["point"]
    values = {k: decompose(a[f"{k}_vs_cv"], a[f"{k}_vs_{job.OLS}"], gap)
              for k in (*study.BASE, *study.WHOLE)}
    run.once(study.OUT / "debt.json", run.stable(dict(analysis=run.artifact(p),
        code=run.artifact(Path(__file__)), sum_debit_floor_pp=gap,
        aggregate_triangle_bound_is_necessary_not_sufficient=True,
        conservative_protocol_unchanged=True, values=values,
        no_additional_labels_read=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
