"""Evaluate support-first scheduling without relaxing the recording risk rule."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_record_support as record
from src.world_model import m3w_opendd_support_schedule as model

admit, stop, run, two, hard, v, base, prior = (
    record.admit, record.stop, record.run, record.two, record.hard, record.v, record.base, record.prior)
OUT = record.OUT.parent/"opendd_support_schedule_v1"
ARMS = ("gain", "independent", "anchor_first", "anchor_matched", "learned_anchor", "learned_matched")
MATCH = (("anchor_first", "anchor_matched"), ("learned_anchor", "learned_matched"))
BOUNDS = ("old_budget_any_order", "old_budget_support_first", "anchor_budget_any_order",
          "anchor_budget_support_first", "any_up_to_two")


def register():
    key, outer, old, source, gain, prev, _, _, capacity = admit.register()
    paths = [admit.OUT/f"{k}.json" for k in ("analysis", "head_freeze")]
    paths += [record.OUT/"analysis.json", stop.OUT/"menu_capacity.json"]
    a, h, b, c = map(run.read, paths)
    if (a["registration"] != key or h["registration"] != key or a["heads"] != run.sha(paths[1])
            or b["heads"] != run.sha(record.OUT/"head_freeze.json") or c != capacity
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (a, b, c))):
        raise ValueError("Require matched development parents")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(p) for p in paths], native_steps=[8, 12], seeds=two.study.prior.SEEDS,
        original_2pct=True, confirmation_access=False, no_new_training=True, no_threshold_search=True,
        policies="support before or on same query as gain; original amplitudes and one slot per role",
        learned="frozen v95 local support head; easy and local-support thresholds both .5",
        primary="anchor-first versus same-support matched earliest gain, preserving exact per-record role counts",
        timing_control="retrospective paired support identity and gain eligibility only; not a deployable arm",
        bounds="exact outcomes-only search across both roles at the original or new realized budget",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), outer, old, source, gain, prev, h["heads"], capacity


@torch.inference_mode()
def evaluate(d, bank, path, release, gain_head, local_head, cuts, ref, previous):
    cv, ols = bank[:2]
    small, x, _, easy, blocks = stop.inputs(d, cv, path, release)
    large = cv+base.model.GAIN_SCALE*(path-cv)
    f, scale = prior.feature(d, cv, large)
    gain = gain_head(torch.from_numpy(f)).numpy()[:, 0]*scale
    y, valid, q, rec = (d[k] for k in ("y", "valid", "query", "rec"))
    pred, plans = {k: cv.copy() for k in ARMS}, {}
    uses = {k: Counter() for k in ARMS}
    for r, seq in blocks.items():
        a, b, _ = base.model.choose((z, gain[z], easy[z]) for z in seq)
        packets = []
        for z in seq:
            rows, anchor = admit.model.menu(z, easy)
            prob = np.zeros(len(z))
            if anchor is not None:
                prob = local_head(torch.from_numpy(admit.model.features(x, easy, rows, anchor)))[:, 1].sigmoid().numpy()
            packets.append((z, gain[z], easy[z], prob))
        output = {"gain": (a, None), "independent": (a, b)}
        for k, m in MATCH:
            j, i = model.stream(iter(packets), learned=k == "learned_anchor")
            early = model.earliest_gain(iter(packets), excluded=i) if j is not None else None
            if j is not None and (early is None or q[j] < q[i] or q[early] > q[j]):
                raise ValueError("Invalid matched role schedule")
            output[k], output[m] = (j, i), (early, i)
            uses[k].update(delayed_gain=int(j is not None and q[early] < q[j]),
                           different_gain=int(j != early))
        for k, (j, i) in output.items():
            if j is not None:
                pred[k][j] = large[j]
            if i is not None:
                if i == j:
                    raise ValueError("Two roles share one row")
                pred[k][i] = small[i]
            uses[k].update(gain=int(j is not None), support=int(i is not None),
                           both=int(j is not None and i is not None))
        plans[r] = output
    metrics = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    for k, old in (("gain", "gain"), ("independent", "two_roles")):
        if any(metrics[k][f] != ref["metrics"][old][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen control did not replay")
    for k, m in MATCH:
        if any(uses[k][f] != uses[m][f] for f in ("gain", "support", "both")):
            raise ValueError("Matched comparison changed role counts")
    for k in ARMS:
        if metrics[k]["edits"] != uses[k]["gain"]+uses[k]["support"]:
            raise ValueError("Scored intervention count changed")
    s = record.contributions(cv, ols, small, y, valid, cuts[0])
    l = record.contributions(cv, ols, large, y, valid, cuts[0])
    counts = Counter({k: 0 for k in BOUNDS})
    legal = (small != cv).any((1, 2))
    for r, output in plans.items():
        ix = np.flatnonzero(rec == r)
        easy_mass, offset = s[2][ix].sum(), s[4][ix].sum()
        counts["no_full_easy_reference"] += int(easy_mass <= 0.)
        counts["no_legal_easy_recipient"] += int(not (legal[ix] & (s[2][ix] > 0.)).any())
        orig = tuple(int(i is not None) for i in output["independent"])
        new = tuple(int(i is not None) for i in output["anchor_first"])
        big, sm = tuple(x[ix] for x in l[:4]), tuple(x[ix] for x in s[:4])
        d0 = dict(y=y[ix], valid=valid[ix], rec=rec[ix])

        def feasible(n, ordered):
            pair = model.witness(big, sm, q[ix], legal[ix], n, ordered, easy_mass, offset)
            if pair is None:
                return False
            j, i = pair
            p = cv[ix].copy()
            if j is not None:
                p[j] = large[ix[j]]
            if i is not None:
                p[i] = small[ix[i]]
            z = v.parent.score(d0, cv[ix], ols[ix], p, cuts)
            if z["dual_positive"] != 1 or z["edits"] != sum(n):
                raise ValueError("Outcome-aware capacity witness failed the actual scorer")
            return True

        bounds = [feasible(orig, False), feasible(orig, True), feasible(new, False), feasible(new, True)]
        bounds.append(bounds[0] or bounds[2] or any(feasible(n, False) for n in ((1, 1), (1, 0), (0, 1))))
        if bounds[0] < bounds[1] or bounds[2] < bounds[3]:
            raise ValueError("An order restriction cannot add capacity")
        counts.update(dict(zip(BOUNDS, bounds, strict=True)))
    if (counts["old_budget_any_order"] < previous["fixed"]["all_legal"]
            or counts["old_budget_any_order"] < previous["any_query"]["all_legal"]
            or counts["anchor_budget_support_first"] < metrics["anchor_first"]["dual_positive"]):
        raise ValueError("Exact capacity did not recover feasible subsets")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+list(MATCH)+[("anchor_first", "independent"), ("learned_anchor", "anchor_first")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
               for a, b in pairs}
    return dict(metrics=metrics, effects=effects, uses=uses, capacity=counts)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    key, outer, old, source, gains, prev, local, capacity = register()
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    result = {}
    for s, d in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for seed in two.study.prior.SEEDS:
            name = f"{s}_{seed}"
            release = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            lookup = lambda a: next(z for z in a["by_site"][s] if z["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            g = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            h = two.fit.load(local[name], lambda: admit.model.network(2*n+3), admit.STEPS)
            cell = evaluate(d, bank, path, release, g, h, cut, lookup(prev), lookup(capacity)["counts"])
            result[s].append(dict(seed=seed, **cell))
            print(s, seed, "support schedule evaluated", flush=True)
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        uses={k: sum((r["uses"][k] for r in flat), Counter()) for k in ARMS},
        capacity=sum((r["capacity"] for r in flat), Counter()))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, by_site=result, summary=summary,
        original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
