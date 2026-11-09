"""Attribute the frozen development selections to the utility and risk checks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recording_utility_guard as run_stage


run, np, stage, study = run_stage.run, run_stage.np, run_stage.stage, run_stage.study


def main():
    key, prior_key, _, _, roster, seal = run_stage.register()
    p = run_stage.PUBLIC/"analysis.json"
    ref = run.read(p)
    frozen = run.read(run.checked(ref["freeze"]))
    if ref["registration"] != key or frozen["registration"] != key or ref["confirmation_access"] or frozen["future_labels_read"]:
        raise ValueError("Require frozen development rows")
    (_, h, risks, d, groups, passed, dual), _ = stage.read_frozen(prior_key, roster, seal)
    point, envelope = run_stage.risk_guard.relative_bounds(h[..., None], risks, groups)
    hist, rows = [], []
    for f, a in zip(seal["recordings"], frozen["recordings"], strict=True):
        r = run.read(stage.parent.PRIVATE/f"{f['recording']}.score.json")
        if r["freeze"] != f or a["recording"] != f["recording"] or a["choices"] != f["choices"]:
            raise ValueError("Scored forecasts changed")
        hist.extend(a["gain_pp"])
        rows.extend([row[0] for row in r["observed"]])
    donor = d[:, 0]*100.
    x = np.column_stack((donor, hist))
    y = np.array([r["actual"]["ols4"] for r in rows])
    out = {}
    for name, data in (("history", x[:, 1:]), ("combined", x)):
        _, lower = study.folds(data, y, groups)
        masks = dict(utility_only=lower > 0.)
        for r_name, values in (("risk_point", point), ("risk_log_envelope", envelope)):
            gate = (values[:, 0] <= stage.pool.TOLERANCE).all(1)
            masks[r_name+"_and_utility"] = gate & (lower > 0.)
            masks[r_name+"_and_raw_donor"] = gate & (donor > 0.)
        m = {}
        for k, a in masks.items():
            sources = np.unique(groups[a])
            per_source = {}
            for g in sources:
                use = np.flatnonzero(a & (groups == g))
                z = stage.parent.prior.merge_comparisons([rows[i]["masses"]["ols4"] for i in use])
                per_source[str(g)] = dict(views=len(use), risk_pass=int(passed[use, 0].sum()),
                    both_pass=int(dual[use, 0].sum()), lower_vs_ols4_pp=z["completion_lower_pp"])
            m[k] = dict(selected=int(a.sum()), risk_pass=int(passed[a, 0].sum()), both_pass=int(dual[a, 0].sum()), by_source=per_source)
            if k.endswith("_and_utility"):
                main = ref["metrics"][name]["gates"][k.removesuffix("_and_utility")]["utility_lower"]
                if [m[k][j] for j in ("selected", "risk_pass", "both_pass")] != [main[j] for j in ("selected", "passed_risk", "passed_both")]:
                    raise ValueError("Selection count did not reproduce")
        out[name] = m
    run.once(run_stage.PUBLIC/"attribution.json", run.stable(dict(source=run.artifact(p), code=run.artifact(Path(__file__)),
        metrics=out, development_followup=True, original_2pct=True, confirmation_access=False)))
    print(out, flush=True)


if __name__ == "__main__":
    main()
