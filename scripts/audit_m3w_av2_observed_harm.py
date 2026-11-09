"""Check conditional risk calibration without reopening raw future outcomes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_observed_harm as run

study = run.study


def main():
    key, _, _, _, _, _, _, _, parent, _, _, _ = run.register()
    path = run.OUT/"summary.json"
    s = study.read(path)
    f = study.read(study.checked(s["freeze"]))
    if f["registration"] != key or f["held_labels_used"] or s["reserved_outcomes_opened"]:
        raise ValueError("Require frozen scored development heads")
    rows, cells = [], []
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        p = z["fit"]["observed_harm"]/z["fit"]["supported"]
        source = study.read(run.support.LOCAL/f"{name}.support.json")
        if source["parent"] != parent["folds"][name]:
            raise ValueError("Wrong support annotation")
        metrics = list(run.support.prior.role_rows(z, parent["folds"][name], name, "held"))
        supported = []
        for a, b, m in zip(z["rows"]["held"], source["cells"], metrics, strict=True):
            if a["clip"] != b["clip"]:
                raise ValueError("Wrong score alignment")
            if b["support"]:
                rows.append(dict(clip=a["clip"], city=a["city"], p=a["harm_probability"], prior=p,
                                 y=b["cause"] == "observed_positive_harm"))
                supported.append(dict(m, support=True, harm=b["cause"] == "observed_positive_harm"))
        budget = min(5, len(supported))
        if budget:
            cells.append(dict(city=z["identity"]["held"], clip="known_support", n=budget, arms=run.cells(supported, budget)))
    if len(rows) != s["support_views"] or sum(r["y"] for r in rows) != s["observed_harm_views"]:
        raise ValueError("Wrong supported cohort")
    brier = lambda r, k: (r[k]-r["y"])**2
    bands = []
    for i in range(5):
        lo, hi = i/5., (i+1)/5.
        z = [r for r in rows if lo <= r["p"] < hi or i == 4 and r["p"] == 1.]
        bands.append(dict(lo=lo, hi=hi, rows=len(z), observed=sum(r["y"] for r in z),
                          mean_probability=sum(r["p"] for r in z)/len(z) if z else None))
    d = dict(source=study.artifact(path), conditional_views=len(rows), risk_band_counts=bands,
        fit_brier=run.av.interval(rows, lambda r: brier(r, "p")),
        donor_rate_brier=run.av.interval(rows, lambda r: brier(r, "prior")),
        brier_delta=run.av.interval(rows, lambda r: brier(r, "p")-brier(r, "prior")),
        used_for_selection=False, independent_confirmation=False, reserved_outcomes_opened=False,
        result_source="fresh_run", outcome_source="cached_verified")
    run.base.immutable(run.OUT/"conditional_calibration.json", d)
    run.base.immutable(run.OUT/"supported_only_ranks.json", dict(source=study.artifact(path),
        **run.summarize_cells(cells), known_support_is_outcome_only=True, used_for_selection=False,
        independent_confirmation=False, reserved_outcomes_opened=False, result_source="fresh_run"))
    print(study.json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
