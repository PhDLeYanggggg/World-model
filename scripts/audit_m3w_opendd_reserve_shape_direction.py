"""Recover both lower-bound directions and check the reserve-only signal."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_reserve_shape as shape

run, v = shape.run, shape.v
SINGLE = shape.OUT.parent/"opendd_reserve_only_v1"


def ci(x):
    x = np.asarray(x)
    draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
    return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())


def main():
    key, _, _, source, *_ = shape.register()
    p, b = shape.OUT/"analysis.json", SINGLE/"analysis.json"
    a, o = run.read(p), run.read(b)
    reg = SINGLE/"registration.json"
    if (a["registration"] != key or o["registration"] != run.sha(reg)
            or run.read(reg)["parent"]["sha256"] != run.sha(p)
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, o))):
        raise ValueError("Require frozen shape and reserve-only evidence")
    f = shape.OUT/"direction_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p), run.artifact(b)],
        original_2pct=True, confirmation_access=False, no_new_policy=True,
        derivation="reverse lower = forward lower minus twice the observed gain; debit symmetric",
        post_hoc_site_sensitivity=True)))
    sites = sorted(a["by_site"])
    mass, reverse, single = {}, {k: {} for k in shape.CLOCKS}, {}
    for s in sites:
        d = study.c.forecast.load(s, source)[0]
        cv = v.parent.prior.analytic(d)[0]
        mass[s] = np.linalg.norm(cv-d["y"], axis=-1)[d["valid"]].sum()
        if mass[s] <= 0:
            raise ValueError("Expected positive CV error mass at every site")
        single[s] = float(np.mean([r["effects"]["reserve_free_vs_cv"] for r in o["by_site"][s]]))
        for k in shape.CLOCKS:
            values = []
            for cell in a["by_site"][s]:
                z = cell[k]
                g = 100.*(z["error_mass"]["neural_matched"]["ade_mass"]-z["error_mass"]["ols4_matched"]["ade_mass"])/mass[s]
                forward = z["effects"]["ols4_matched_vs_neural_matched"]
                if forward > g+1e-12:
                    raise ValueError("Conservative gain exceeds observed gain")
                ans = forward-2.*g
                if k == "gain":
                    q = next(r for r in o["by_site"][s] if r["seed"] == cell["seed"])
                    np.testing.assert_allclose(ans,
                        q["effects"]["reserve_neural_matched_vs_reserve_ols_matched"], atol=1e-12, rtol=0.)
                values.append(ans)
            reverse[k][s] = float(np.mean(values))
    x = list(single.values())
    run.once(shape.OUT/"direction.json", run.stable(dict(registration=run.sha(f), original_2pct=True,
        confirmation_access=False, neural_vs_ols4={k: ci(list(values.values())) for k, values in reverse.items()},
        reserve_free=dict(site=single, mass_weighted_pp=float(np.average(x, weights=list(mass.values()))),
                          leave_one_out={s: ci(x[:i]+x[i+1:]) for i, s in enumerate(sites)}))))


if __name__ == "__main__":
    main()
