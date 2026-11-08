"""Bound second-slot timing at the same recordings, count and amplitude."""

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

v, run, hard, two, prior, OUT = base.v, base.run, base.hard, base.two, base.prior, base.OUT


@torch.inference_mode()
def evaluate(d, bank, path, release, net, cut, ref):
    cv, ols = bank[:2]
    large = cv+base.model.GAIN_SCALE*(path-cv)
    small = cv+base.model.SUPPORT_SCALE*(path-cv)
    x, scale = prior.feature(d, cv, large)
    gain = net(torch.from_numpy(x)).numpy()[:, 0]*scale
    easy = release(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()[:, 0]
    q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
    legal = (large != cv).any((1, 2))
    counts = Counter()
    for r in np.unique(rec):
        at = np.flatnonzero(rec == r)
        packets = [at[(q[at] == j) & legal[at]] for j in np.unique(q[at])]
        a, b, _ = base.model.choose((ix, gain[ix], easy[ix]) for ix in packets)
        p = cv[at].copy()
        if a is not None:
            p[np.searchsorted(at, a)] = large[a]
        d0 = dict(y=y[at], valid=valid[at], rec=rec[at])
        args = cv[at], ols[at]
        if b is None:
            t = v.parent.score(d0, *args, p, cut)["dual_positive"]
            counts.update(no_slot=1, no_slot_joint=t, frozen_query_joint=t, any_query_joint=t)
            continue
        candidates = small[at]
        only_query = np.where((q[at] == q[b])[:, None, None], candidates, cv[at])
        one = base.best_second(*args, only_query, y[at], valid[at], p, cut[0])
        any_time = base.best_second(*args, candidates, y[at], valid[at], p, cut[0])
        if one is not None and any_time is None:
            raise ValueError("Expanded timing cannot lose a witness")
        for label, j in (("frozen_query_joint", one), ("any_query_joint", any_time)):
            z = p.copy()
            i = int(np.searchsorted(at, b)) if j is None else j
            z[i] = candidates[i]
            score = v.parent.score(d0, *args, z, cut)
            if (score["edits"] != int(a is not None)+1 or (j is not None and score["dual_positive"] != 1)):
                raise ValueError("Timing witness did not preserve count and original risk")
            counts[label] += score["dual_positive"]
        counts.update(has_slot=1, needs_other_query=int(one is None and any_time is not None))
    if counts["frozen_query_joint"] != ref["metrics"]["oracle_same_query"]["dual_positive"]:
        raise ValueError("Fixed-query oracle changed")
    return counts


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, heads, _ = base.register()
    p = OUT/"analysis.json"
    a = run.read(p)
    if a["registration"] != key or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require verified two-slot parent")
    f = OUT/"timing_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(p), original_2pct=True,
        confirmation_access=False, outcome_aware_only=True, no_new_training=True,
        permitted_change="only the second recipient and query may change; same recording coverage, first action and amplitude")))
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
            result[s].append(dict(seed=seed, counts=evaluate(d, bank, path, release, net, cut, ref)))
        print(s, "matched second-slot timing bound complete", flush=True)
    total = sum((r["counts"] for rows in result.values() for r in rows), Counter())
    run.once(OUT/"timing_capacity.json", run.stable(dict(registration=run.sha(f), summary=total, by_site=result,
        outcome_aware_only=True, original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
