"""Separate the CV-centered radius limit from failures of learned path choice."""

from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_hannover_frozen_transfer as study

run, np, torch, source = study.run, study.np, study.torch, study.source


def capacity(cv, radius, y, valid, cut):
    error = np.linalg.norm(cv-y, axis=-1)
    mass = error[valid].sum()
    known = valid.any(1)
    ade = np.where(valid, error, 0.).sum(1)/valid.sum(1).clip(1)
    easy = (radius > 0.) & known & (ade > 0.) & (ade <= cut)
    upper = np.minimum(error, radius[:, None])[valid].sum()
    easy_mass = ade[easy].sum()
    unknown_debit = radius[~known].sum()
    return dict(optimistic_utility_upper_pp=float(100*upper/mass),
                unknown_only_easy_risk_lower_pct=float(100*unknown_debit/easy_mass) if easy_mass > 0. else None,
                easy_rows=int(easy.sum()))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, _, _, scale, _ = study.register()
    p = study.OUT / "analysis.json"
    old = run.read(p)
    if old["registration"] != key or old["confirmation_access"] or not old["no_target_tuning"]:
        raise ValueError("Require the frozen transfer before calculating hindsight limits")
    r = study.OUT / "capacity_registration.json"
    run.once(r, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(p),
        uses_labels_only_for_diagnostic=True, original_2pct=True, confirmation_access=False)))
    nets = study.transfer.frozen_models(*frozen[:5])
    cut = frozen[-1]["open_dd_positive_complete_cv_cuts"][0]
    out = {}
    with ZipFile(study.ZIP) as z:
        for clip in study.CLIPS:
            grid, _, _ = study.parse_csv(z.read(study.PREFIX+f"{clip}_trajectory.csv"))
            h, y, _ = study.windows(grid, scale)
            valid = np.isfinite(y).all(2)
            out[clip] = []
            for i, seed in enumerate(study.choice_study.SEEDS):
                cv, _, base, _, _ = study.transfer.forecast(h, nets[seed])
                radius = np.linalg.norm(base-cv, axis=-1).max(1)
                a = old["by_clip"][clip]["scores"][i]
                if a["seed"] != seed:
                    raise ValueError("Frozen seed order changed")
                values = {}
                for part, f in {"original": 1., **study.transfer.FRACTIONS}.items():
                    s = capacity(cv, f*radius, y, valid, cut)
                    v = a[part if part in a else "half"]
                    s["ols4_conservative_pp"] = v["ols4"]["lower_pp"]["cv"]
                    s["observed_ols4_pp"] = v["ols4"]["known_pp_vs_cv"]
                    if part in a:
                        if v["oracle_utility"]["lower_pp"]["cv"] > s["optimistic_utility_upper_pp"]+1e-10:
                            raise ValueError("Matched-bank utility exceeds its enclosing radius")
                        if (v["oracle_min_risk"]["risk"]["completion_easy_risk_upper"]*100+1e-10
                                < s["unknown_only_easy_risk_lower_pct"]):
                            raise ValueError("Reported risk falls below unavoidable unknown debit")
                    values[part] = s
                out[clip].append(dict(seed=seed, **values))
    summary = {}
    for p in ("original", *study.transfer.FRACTIONS):
        rows = [v[p] for x in out.values() for v in x]
        summary[p] = dict(views=len(rows), mean_optimistic_upper=float(np.mean([r["optimistic_utility_upper_pp"] for r in rows])),
            mean_ols4_lower=float(np.mean([r["ols4_conservative_pp"] for r in rows])),
            ols4_lower_exceeds_optimistic_upper=sum(r["ols4_conservative_pp"] > r["optimistic_utility_upper_pp"] for r in rows),
            unknown_alone_exceeds_easy_limit=sum(r["unknown_only_easy_risk_lower_pct"] > 2. for r in rows))
    run.once(study.OUT / "capacity.json", run.stable(dict(registration=run.sha(r), summary=summary, by_clip=out,
        risk_lower_is_not_certificate=True, no_training=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
