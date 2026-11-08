"""Decompose missed expanded-menu opportunities at the frozen trained policy."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import run_m3w_opendd_support_directions as study

run, v, two, hard, prior, early, wait, old = study.run, study.v, study.two, study.hard, study.prior, study.early, study.wait, study.old
OUT = study.OUT
MISSES = ("no_release", "wrong_query", "wrong_recipient", "wrong_direction", "realized")


@torch.inference_mode()
def audit(d, menu, groups, net, cutoff, metrics):
    x = menu[2]
    z = net(torch.from_numpy(x.reshape(-1, x.shape[-1]))).numpy().reshape(*x.shape[:2], 2)
    q = d["query"]
    counts = Counter({k: 0 for k in (*MISSES, "no_release_utility", "no_release_loss_guard",
                                     "original_joint_destroyed", "new_opportunities")})
    for _, _, _, _, rows, kinds, _, feasible, original_joint in groups:
        scores = z[kinds, rows]
        choice = study.model.first_release(rows, kinds, q[rows], scores[:, 0], scores[:, 1] < 0., cutoff)
        if choice is None:
            if not original_joint and feasible.any():
                counts["new_opportunities"] += 1
                counts["no_release"] += 1
                reason = "no_release_loss_guard" if (scores[feasible, 0] > cutoff).any() else "no_release_utility"
                counts[reason] += 1
            continue
        row, family = choice
        selected = (rows == row) & (kinds == family)
        new_joint = bool((feasible & selected).any())
        if original_joint:
            counts["original_joint_destroyed"] += int(not new_joint)
            continue
        if not feasible.any():
            continue
        counts["new_opportunities"] += 1
        reason = ("realized" if new_joint else "wrong_direction" if (feasible & (rows == row)).any()
                  else "wrong_recipient" if (feasible & (q[rows] == q[row])).any() else "wrong_query")
        counts[reason] += 1
    base = metrics["original"]["dual_positive"]
    if (base+counts["new_opportunities"] != metrics["window_expanded"]["dual_positive"]
            or base+counts["realized"]-counts["original_joint_destroyed"] != metrics["expanded"]["dual_positive"]
            or sum(counts[k] for k in MISSES) != counts["new_opportunities"]):
        raise ValueError("Miss attribution did not reconcile with exact evaluation")
    return counts


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old_forecasts, source, gains, *_, frozen, frozen_heads = study.register()
    a = early.timing.legacy.read(OUT.name)
    h = run.read(OUT/"head_freeze.json")
    if a["registration"] != key or h["registration"] != key:
        raise ValueError("Require frozen expanded-menu analysis")
    f = OUT/"attribution_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(OUT/f"{k}.json")
        for k in ("registration", "analysis", "head_freeze")], original_2pct=True, confirmation_access=False,
        no_new_training=True, diagnostic_only=True,
        breakdown="missed joint pass among originally failing views: no trigger, clock, recipient, shape")))
    sites = {s: wait.study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    result = {}
    for s, d in sites.items():
        cv, ols = v.parent.prior.analytic(d)[:2]
        cuts = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for seed in two.study.prior.SEEDS:
            name = f"{s}_{seed}"
            lookup = lambda a: next(z for z in a["by_site"][s] if z["seed"] == seed)
            release = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(lookup(old_forecasts)["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
            first = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            p = early.prepare(d, cv, path, release, first)
            h0 = two.fit.load(frozen_heads[name], lambda: early.model.network(p[-2].shape[1]), old.STEPS)
            roster = study.ref.proposals(p, h0, lookup(frozen)["donor_threshold"][0])
            m = study.menu(d, cv, ols, p)
            groups = study.cases(d, cv, ols, cuts, p, m, roster)
            net = two.fit.load(h["heads"][name+"_expanded"], lambda: study.model.network(m[2].shape[-1]), study.STEPS)
            ref = lookup(a)
            result[s].append(dict(seed=seed, counts=audit(d, m, groups, net, ref["cutoffs"]["expanded"], ref["metrics"])))
        print(s, "miss attribution audited", flush=True)
    total = sum((z["counts"] for group in result.values() for z in group), Counter())
    run.once(OUT/"attribution_analysis.json", run.stable(dict(registration=run.sha(f), summary=total,
        by_site=result, original_2pct=True, confirmation_access=False, diagnostic_only=True)))
    print(run.stable(total), flush=True)


if __name__ == "__main__":
    main()
