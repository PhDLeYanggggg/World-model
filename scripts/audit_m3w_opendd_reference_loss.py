"""Exact outcome-aware thinning of the frozen early-support proposal set."""

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
from scripts import run_m3w_opendd_reference_loss as study

run, v, two, hard, prior, old, early, wait = study.run, study.v, study.two, study.hard, study.prior, study.old, study.early, study.wait
OUT = study.OUT
ARMS = ("exact_joint", "no_loss_at_most", "no_loss_utility_at_most")


def cell(d, cv, ols, cuts, pack, roster, ref):
    large, small = pack[:2]
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    base = cv.copy()
    frozen = cv.copy()
    changed = []
    for r, g, b, j in roster:
        if g is not None:
            base[g] = frozen[g] = large[g]
        if b is not None:
            base[b] = small[b]
        if j is not None:
            frozen[j] = small[j]
        if j != b and j is not None:
            changed.append((r, b, j))
    original = v.parent.score(d, cv, ols, base, cuts)
    full = v.parent.score(d, cv, ols, frozen, cuts)
    for k, z in (("original", original), ("frozen", full)):
        if any(z[f] != ref["metrics"][k][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen support replay disagrees with the registered scorer")

    risk = [early.policy(p, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]
            for p in (base, frozen)]
    gain = [np.stack([v.paired(b, p, y, valid)[1].sum(1) for b in (cv, ols)]) for p in (base, frozen)]
    delta = v.paired(base, frozen, y, valid)[1].sum(1)
    options = []
    for r, b, j in changed:
        ix = rec == r
        a, c = (z[r] for z in risk)
        before, after = (int(z["supported"] and (g[:, ix].sum(1) > 0.).all())
                         for z, g in zip((a, c), gain, strict=True))
        lost = a["completion_easy_risk_upper"] is not None and c["completion_easy_risk_upper"] is None
        options.append((r, b, j, after-before, float(delta[ix].sum()), bool(lost)))
    if sum(z[-1] for z in options) != ref["diagnostics"].get("frozen_lost_reference", 0):
        raise ValueError("Reference-loss attribution changed")

    count = Counter()
    pred = {k: base.copy() for k in ARMS}
    quotas = [ref["diagnostics"].get("direct_"+k, 0) for k in ("added", "replaced")]
    for add, quota in zip((True, False), quotas, strict=True):
        pool = [z for z in options if (z[1] is None) == add]
        safe = [z for z in pool if not z[-1]]
        count["no_loss_quota_shortfall"] += max(0, quota-len(safe))
        arms = {"exact_joint": sorted(pool, key=lambda z: (-z[3], -z[4]))[:quota],
                "no_loss_at_most": sorted((z for z in safe if (z[3], z[4]) > (0, 0.)),
                                          key=lambda z: (-z[3], -z[4]))[:quota],
                "no_loss_utility_at_most": sorted((z for z in safe if z[4] > 0.),
                                                  key=lambda z: -z[4])[:quota]}
        for k, picks in arms.items():
            for _, b, j, joint, _, lost in picks:
                if b is not None:
                    pred[k][b] = cv[b]
                pred[k][j] = small[j]
                count.update({k+"_added": int(add), k+"_replaced": int(not add),
                              k+"_joint_delta": joint, k+"_lost_reference": int(lost)})
    stats = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    for k, z in stats.items():
        if (z["dual_positive"] != original["dual_positive"]+count[k+"_joint_delta"]
                or z["edits"] != original["edits"]+count[k+"_added"]):
            raise ValueError("Outcome-aware witnesses did not rescore")
    if (count["exact_joint_added"] != quotas[0] or count["exact_joint_replaced"] != quotas[1]
            or stats["exact_joint"]["dual_positive"] < ref["metrics"]["direct"]["dual_positive"]):
        raise ValueError("The exact count menu must contain the direct policy")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {k+"_vs_original": hard.paired_pp(base, p, y, valid, mass) for k, p in pred.items()}
    return dict(metrics=stats, effects=effects, diagnostics=count)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old_forecasts, source, gains, *_, frozen, heads = study.register()
    ref = early.timing.legacy.read(OUT.name)
    if ref["registration"] != key:
        raise ValueError("Require the frozen loss-guard study")
    f = OUT/"oracle_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(OUT/f"{k}.json")
        for k in ("registration", "head_freeze", "analysis")], original_2pct=True, confirmation_access=False,
        outcome_aware_only=True, budget="direct gate additions and replacements per location and seed",
        options="freeze each early proposal and original forecast; select exact quota or at most quota without reference loss")))
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
            head = two.fit.load(heads[name], lambda: early.model.network(p[-2].shape[1]), old.STEPS)
            roster = study.proposals(p, head, lookup(frozen)["donor_threshold"][0])
            result[s].append(dict(seed=seed, **cell(d, cv, ols, cuts, p, roster, lookup(ref))))
        print(s, "exact support thinning audited", flush=True)
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
    run.once(OUT/"oracle_analysis.json", run.stable(dict(registration=run.sha(f), by_site=result, summary=summary,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))
    print(run.stable(summary), flush=True)


if __name__ == "__main__":
    main()
