"""Hold support timing fixed and rank gain within the same easy eligibility."""

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
from scripts import run_m3w_opendd_two_reserves as base

v, run, hard, two, prior = base.v, base.run, base.hard, base.two, base.prior
OUT = base.OUT
ARMS = ("gain", "easy", "constrained_gain", "gain_matched")


def rerank(rows, gain, easy, gain_recipient):
    take = (rows != gain_recipient) & (easy >= base.model.EASY_THRESHOLD)
    if not take.any():
        raise ValueError("The frozen support query must contain an eligible recipient")
    return int(rows[np.where(take, gain, -np.inf).argmax()])


@torch.inference_mode()
def evaluate(d, bank, path, release, net, cut, ref):
    cv, ols = bank[:2]
    large = cv+base.model.GAIN_SCALE*(path-cv)
    small = cv+base.model.SUPPORT_SCALE*(path-cv)
    x, scale = prior.feature(d, cv, large)
    gain = net(torch.from_numpy(x)).numpy()[:, 0]*scale
    easy = release(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()[:, 0]
    q, rec, y, mask = (d[k] for k in ("query", "rec", "y", "valid"))
    legal = (large != cv).any((1, 2))
    out = {k: cv.copy() for k in ARMS}
    changes = Counter()
    truth = two.moments(cv, cv, y, mask, cut[0])["easy"]
    for r in np.unique(rec):
        local = rec == r
        packets = [np.flatnonzero(local & legal & (q == j)) for j in np.unique(q[local])]
        a, b, c = base.model.choose((at, gain[at], easy[at]) for at in packets)
        if a is not None:
            for p in out.values():
                p[a] = large[a]
        if b is not None:
            rows = np.flatnonzero(local & legal & (q == q[b]))
            j = rerank(rows, gain[rows], easy[rows], a)
            if a in (j, b, c) or q[j] != q[b] or q[c] != q[b]:
                raise ValueError("Ranking altered the support time or first recipient")
            out["easy"][b], out["constrained_gain"][j], out["gain_matched"][c] = small[b], small[j], small[c]
            changes.update(extras=1, different=int(j != b), easy=int(truth[j]), original_easy=int(truth[b]))
    metrics = {k: v.parent.score(d, cv, ols, p, cut) for k, p in out.items()}
    for k, earlier in (("gain", "gain"), ("easy", "two_roles"), ("gain_matched", "two_gain_matched")):
        if any(metrics[k][f] != ref["metrics"][earlier][f] for f in v.prior.FIELDS):
            raise ValueError("Same-query control did not reproduce")
    if metrics["constrained_gain"]["edits"] != metrics["easy"]["edits"]:
        raise ValueError("Ranking changed count")
    mass = np.linalg.norm(cv-y, axis=-1)[mask].sum()
    effects = {k: hard.paired_pp(cv if k == "cv" else out[k], out["constrained_gain"], y, mask, mass)
               for k in ("cv", "gain", "easy", "gain_matched")}
    return dict(metrics=metrics, effects=effects, support=changes)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, heads, _ = base.register()
    f = OUT/"analysis.json"
    a = run.read(f)
    if a["registration"] != key or not a["original_2pct"] or a["confirmation_access"]:
        raise ValueError("Require frozen two-role policies")
    p = OUT/"ranking_registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(f), no_new_training=True,
        post_parent_development_followup=True, original_2pct=True, confirmation_access=False, no_threshold_search=True,
        primary="gain rank among predicted easy versus easy rank, identical query and amplitude",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, _) in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            release = two.fit.load(outer[f"{s}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            net = two.fit.load(heads[f"{s}_{seed}"], lambda: prior.model.network(n), prior.STEPS)
            ref = next(r for r in a["by_site"][s] if r["seed"] == seed)
            result[s].append(dict(seed=seed, **evaluate(d, bank, path, release, net, cut, ref)))
        print(s, "same-query ranking complete", flush=True)
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, support=sum((r["support"] for r in flat), Counter()))
    run.once(OUT/"ranking_analysis.json", run.stable(dict(registration=run.sha(p), by_site=result, summary=summary,
        original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
