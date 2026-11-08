"""Test the conjunction of frozen easy and gain signals for the reserve."""

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
from scripts import run_m3w_opendd_reserve as prior

OUT, v, two, run, hard, m = prior.OUT, prior.v, prior.two, prior.run, prior.hard, prior.m
ARMS = ("base", "easy", "gain", "factor", "easy_match", "gain_match")


def factorized(packets):
    for t, (rows, easy, gain) in enumerate(packets):
        eligible = (easy >= prior.model.EASY_THRESHOLD) & (gain > 0.)
        if eligible.any():
            return t, int(rows[np.where(eligible, gain, -np.inf).argmax()])
    return None


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, clock, net, cut, ref):
    cv, ols = bank[:2]
    p = cv+prior.model.AMPLITUDE*(path-cv)
    y, valid, q, rec = (d[k] for k in ("y", "valid", "query", "rec"))
    x, scale = prior.feature(d, cv, p)
    g = net(torch.from_numpy(x)).numpy()[:, 0]*scale
    e = release(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()[:, 0]
    legal = (p != cv).any((1, 2))
    events, detail = {k: [] for k in ARMS[1:]}, {}
    base, state = cv.copy(), {}
    for r, seq in study.packets(d, cv, ols, path, starts, release):
        original, _ = m.stream(iter(seq), lambda x: clock(torch.from_numpy(x)).numpy(), "adaptive_queue")
        ids = np.unique(q[rec == r])
        packets = []
        counts = []
        for j, (used, _, _) in zip(ids, original, strict=True):
            base[used] = path[used]
            at = np.setdiff1d(np.flatnonzero((q == j) & legal), used, assume_unique=True)
            packets.append((at, e[at], g[at]))
            counts.append(len(used))
        state[r] = seq, counts
        for k in ("easy", "gain"):
            z = prior.model.first(((at, easy, gain, easy) for at, easy, gain in packets), k)
            if z is not None:
                events[k].append(z[1])
        z = factorized(iter(packets))
        if z is None:
            continue
        j, row = z
        events["factor"].append(row)
        at, easy, gain = packets[j]
        events["easy_match"].append(int(at[easy.argmax()]))
        events["gain_match"].append(int(at[gain.argmax()]))
    outputs = {"base": base}
    mom = two.moments(cv, p, y, valid, cut[0])
    low = v.paired(cv, p, y, valid)[1].sum(1)
    for k, rows in events.items():
        at = np.asarray(rows, dtype=int)
        if len(np.unique(rec[at])) != len(at) or not (base[at] == cv[at]).all() or not legal[at].all():
            raise ValueError("The factorized arm interfered with the incumbent")
        outputs[k] = base.copy()
        outputs[k][at] = p[at]
        detail[k] = dict(extra=len(at), easy=int(mom["easy"][at].sum()), negative_lower=int((low[at] < 0.).sum()))
    for k in ("easy_match", "gain_match"):
        np.testing.assert_array_equal(q[events["factor"]], q[events[k]])
    stats = {k: v.parent.score(d, cv, ols, z, cut) for k, z in outputs.items()}
    for k in ("base", "easy", "gain"):
        if any(stats[k][f] != ref["metrics"][k][f] for f in v.prior.FIELDS):
            raise ValueError("The reserve parent controls did not reproduce")
    support = mom["easy"]
    cohorts = {k: Counter() for k in prior.prior.GROUPS}
    for r, (seq, counts) in state.items():
        if not sum(counts):
            group = "no_credits"
        elif not any(support[p.rows[j]].any() for p, n in zip(seq, counts, strict=True) if n
                     for j in np.flatnonzero(p.size == n)):
            group = "no_reachable_easy"
        else:
            continue
        at = rec == r
        p = outputs["factor"][at]
        score = v.parent.score(dict(y=y[at], valid=valid[at], rec=rec[at]), cv[at], ols[at], p, cut)
        cohorts[group].update(total=1, extra=int(np.any(p != base[at])), joint=score["dual_positive"])
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    gain = {k: hard.paired_pp(cv if k == "cv" else outputs[k], outputs["factor"], y, valid, mass)
            for k in ("cv", "base", "easy", "gain", "easy_match", "gain_match")}
    return dict(metrics=stats, detail=detail, effects=gain, cohorts=cohorts)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, _, clock, cap = prior.register()
    a, h = (run.read(OUT/f"{k}.json") for k in ("analysis", "head_freeze"))
    if (a["registration"] != key or h["registration"] != key or a["heads"] != run.sha(OUT/"head_freeze.json")
            or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require the frozen reserve study")
    f = OUT/"factor_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(OUT/"analysis.json"),
        source_only_followup=True, confirmation_access=False, original_2pct=True,
        no_new_training=True, no_new_thresholds=True, gate="existing easy >= .50 and gain > 0",
        primary="direct lower gain versus simple easy at exactly the same realized query",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, starts) in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            release = two.fit.load(outer[f"{s}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            p = next(p for _, seq in study.packets(d, *bank[:2], path, starts, release) for p in seq)
            n = p.features.shape[1]+m.QUEUE_WIDTH
            inc = two.fit.load(clock[f"{s}_{seed}_full"], lambda: v.model.network(n), study.STEPS)
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            net = two.fit.load(h["heads"][f"{s}_{seed}"], lambda: prior.model.network(n), prior.STEPS)
            ref = next(r for r in a["by_site"][s] if r["seed"] == seed)
            out = evaluate(d, bank, path, starts, release, inc, net, cut, ref)
            limit = next(r["groups"] for r in cap["by_site"][s] if r["seed"] == seed)
            if any(z["total"] != limit[k].get("total", 0) or z["joint"] > limit[k].get("qualifies_at_0.25", 0)
                   for k, z in out["cohorts"].items()):
                raise ValueError("Factorized arm exceeds the matching source ceiling")
            result[s].append(dict(seed=seed, **out))
        print(s, "factorized reserve complete", flush=True)
    flat = [r for group in result.values() for r in group]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    stats = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS}
    summary = dict(metrics=stats, effects={k: ci(k) for k in flat[0]["effects"]},
        detail={k: sum((Counter(r["detail"][k]) for r in flat), Counter()) for k in ARMS[1:]},
        cohorts={g: sum((r["cohorts"][g] for r in flat), Counter()) for g in prior.prior.GROUPS})
    run.once(OUT/"factor_analysis.json", run.stable(dict(registration=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
