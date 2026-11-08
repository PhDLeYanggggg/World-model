"""Audit sensitivity of the matched dual-reference gain to the site roster."""

from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_candidate_replay as evidence

NAME = "opendd_dual_direction_v1"


def analyze(a):
    groups = a["by_site"]
    if len(groups) != 7 or any(len(v) != 3 for v in groups.values()):
        raise ValueError("Require the original seven sites and three seeds")
    x = np.array([np.mean([r["effects"]["dual_vs_cv_head"] for r in z]) for z in groups.values()])
    if not np.isclose(x.mean(), a["summary"]["effects"]["dual_vs_cv_head"]["point"], rtol=0, atol=1e-12):
        raise ValueError("Site means do not reproduce the primary comparison")
    loo = (x.sum()-x)/(len(x)-1)
    omit = int(x.argmax())
    w = np.delete(x, omit)
    draw = np.random.default_rng(817).integers(len(w), size=(3000, len(w)))
    signs = np.array(list(product([-1., 1.], repeat=len(x))))
    p = float((np.abs(signs @ x) >= abs(x.sum())-1e-12).mean())
    metrics = {}
    for k in ("neural", "cv_head", "dual", "guard", "joint", "fixed_oracle"):
        metrics[k] = {}
        for f in ("vs_cv", "vs_ols4"):
            u = np.array([np.mean([r["metrics"][k][f] for r in z]) for z in groups.values()])
            metrics[k][f] = dict(site_mean=float(u.mean()), worst_site=float(u.min()))
        easy = [r["easy_net_percent"][k] for z in groups.values() for r in z if r["easy_net_percent"][k] is not None]
        metrics[k]["worst_site_seed_easy_net_percent"] = max(easy)
    return dict(site_effects=dict(zip(groups, x.tolist(), strict=True)),
        leave_one_site_out_range=[float(loo.min()), float(loo.max())],
        strongest_site_removed=dict(site=list(groups)[omit], point=float(w.mean()),
                                     interval95=np.quantile(w[draw].mean(1), [.025, .975]).tolist()),
        two_sided_site_sign_flip_p=p, symmetric_site_errors_assumed=True,
        site_roster_exposed=True, no_model_selection=True, reference_and_easy_diagnostics=metrics)


def main():
    p = evidence.REPORTS/NAME
    src = p/"analysis.json"
    a = evidence.read(NAME)
    out = dict(analysis_sha256=evidence.sha(src), original_2pct=True, confirmation_access=False, **analyze(a))
    data = json.dumps(out, indent=2, sort_keys=True, allow_nan=False)+"\n"
    f = p/"sensitivity.json"
    if f.exists():
        if f.read_text() != data:
            raise ValueError("Sensitivity replay changed")
    else:
        with f.open("x") as w:
            w.write(data)
    print(data, end="")


if __name__ == "__main__":
    main()
