"""Check cross-site sensitivity of the frozen same-clock rank experiment."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_rank as study

run, v, OUT = study.run, study.v, study.OUT
KEYS = ("paired_vs_fixed", "guarded_vs_fixed", "paired_vs_scalar", "guarded_vs_cv", "oracle_local_vs_fixed")


def summary(values):
    values = np.asarray(values)
    draw = np.random.default_rng(817).integers(len(values), size=(3000, len(values)))
    return dict(point=float(values.mean()), interval95=np.quantile(values[draw].mean(1), [.025, .975]).tolist())


def main():
    k, _, _, source, *_ = study.register()
    p = OUT/"analysis.json"
    a = run.read(p)
    if (a["registration"] != k or a["confirmation_access"] or not a["original_2pct"]
            or a["heads"] != run.sha(OUT/"head_freeze.json")):
        raise ValueError("Require intact frozen rank evidence")
    f = OUT/"sensitivity_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(p), contrasts=KEYS,
        post_hoc=True, no_model_selection=True, primary_unchanged=True,
        description="leave-one-site sensitivity and observed CV-error-mass weighting; full-roster experiment unchanged",
        original_2pct=True, confirmation_access=False)))
    sites = sorted(a["by_site"])
    mass = []
    for s in sites:
        d = study.study.c.forecast.load(s, source)[0]
        cv = v.parent.prior.analytic(d)[0]
        mass.append(np.linalg.norm(cv-d["y"], axis=-1)[d["valid"]].sum())
    if not np.all(np.isfinite(mass)) or min(mass) <= 0.:
        raise ValueError("All sites require observed CV error mass")
    result = {}
    for k in KEYS:
        x = [np.mean([r["effects"][k] for r in a["by_site"][s]]) for s in sites]
        exact = summary(x)
        for col in ("point", "interval95"):
            np.testing.assert_allclose(exact[col], a["summary"]["effects"][k][col], atol=1e-12, rtol=0.)
        result[k] = dict(site=dict(zip(sites, map(float, x), strict=True)), mass_weighted=float(np.average(x, weights=mass)),
            leave_one_out={s: summary(x[:i]+x[i+1:]) for i, s in enumerate(sites)},
            positive_sites=int(np.count_nonzero(np.array(x) > 0.)),
            nonzero_sites=int(np.count_nonzero(x)))
    run.once(OUT/"sensitivity.json", run.stable(dict(registration=run.sha(f), contrasts=result,
        post_hoc=True, original_2pct=True, confirmation_access=False, submission_ready=False)))
    print(OUT/"sensitivity.json", flush=True)


if __name__ == "__main__":
    main()
