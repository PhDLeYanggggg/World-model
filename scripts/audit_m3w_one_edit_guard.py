"""Secondary donor-only check for a threshold that improves both selection criteria."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_one_edit_stopping as prior

run, parent, two = prior.run, prior.parent, prior.two
OUT = prior.OUT / "donor_guard"


def guard(grid):
    baseline = grid["threshold_0"]
    allowed = [k for k, v in grid.items() if v[0] >= baseline[0] and v[1] >= baseline[1]]
    return max(allowed, key=lambda k: grid[k]), allowed


def execute():
    key, _, _ = prior.register()
    a, h = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze")]
    report, receipt = run.read(a), run.read(h)
    if (any(r["registration"] != key for r in (report, receipt)) or report["heads"] != run.sha(h)
            or report["confirmation_access"] or not report["original_2pct"]):
        raise ValueError("Require the frozen one-edit study")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(a), run.artifact(h)],
        selection="secondary: both donor pass rate and donor CV gain at least immediate; same primary order among feasible thresholds",
        no_training=True, original_2pct=True, confirmation_access=False)))
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    result = {}
    for site, d in sites.items():
        s = report["selected_by_donors"][site]
        choice, allowed = guard(s["stop_grid"])
        cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != site]), [.25, .75])
        packet = two.prepare(d)
        entries = list(prior.records(d, packet))
        rows = []
        for cell in report["by_site"][site]:
            seed = cell["seed"]
            meta = receipt["heads"][f"{site}_{seed}"]
            if meta["identity"] != dict(registration=key, excluded=[site], fit=[k for k in sites if k != site], seed=seed):
                raise ValueError("Outer checkpoint contains incompatible source roles")
            net = two.fit.load(meta, lambda: prior.prior.model.network(packet[2].shape[-1]), prior.STEPS)
            plans = prior.plans(prior.score_windows(packet, entries, net), dict(guard=prior.GRID[choice], immediate=(0, 0.)))
            p, m = prior.forecasts(d, packet, plans, cuts)
            for k, v in cell["metrics"]["immediate"].items():
                np.testing.assert_allclose(m["immediate"][k], v, atol=1e-10, rtol=0.)
            effect = prior.paired_pp(d, packet[0][0], p["guard"], p["immediate"], np.arange(len(p["guard"])))
            rows.append(dict(seed=seed, **m["guard"], lower_pp=effect))
        result[site] = dict(chosen=choice, allowed=allowed, rows=rows)
    fields = ("views", "edits", "dual_positive", "risk_pass", "undefined")
    total = Counter({k: sum(r[k] for v in result.values() for r in v["rows"]) for k in fields})
    x = np.array([np.mean([r["lower_pp"] for r in v["rows"]]) for v in result.values()])
    draws = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
    total["fallback_sites"] = sum(v["chosen"] == "threshold_0" for v in result.values())
    out = dict(registration=run.sha(reg), totals=total, by_site=result,
        lower_vs_immediate_pp=dict(point=float(x.mean()), interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist()),
        original_2pct=True, confirmation_access=False, new_deployment=False)
    run.once(OUT / "analysis.json", run.stable(out))
    print(out["totals"], out["lower_vs_immediate_pp"], flush=True)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    execute()


if __name__ == "__main__":
    main()
