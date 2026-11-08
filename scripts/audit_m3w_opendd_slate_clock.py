"""Measure remaining exact slate capacity at the learned clock."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study

v, run, m = study.v, study.run, study.m
OUT = study.OUT


@torch.inference_mode()
def assess(d, bank, path, starts, release, net, cut):
    cv, ols = bank[:2]
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    moments = study.two.moments(cv, path, y, valid, cut[0])
    risk = np.column_stack([moments[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    before = v.paired(ols, cv, y, valid)[1].sum(1)
    gain = np.c_[v.paired(cv, path, y, valid)[1].sum(1), v.paired(ols, path, y, valid)[1].sum(1)-before]
    cap, learned, witness = Counter(), cv.copy(), cv.copy()
    for r, seq in study.packets(d, cv, ols, path, starts, release):
        plan, _ = m.stream(iter(seq), lambda x: net(torch.from_numpy(x)).numpy(), "adaptive_queue")
        fixed = []
        for p, (chosen, _, _) in zip(seq, plan, strict=True):
            k = len(chosen)
            learned[chosen] = path[chosen]
            # Zero delay over the realized spend stream fixes every query's count.
            rows = (p.rows[0],)+tuple(p.rows[j] for j in np.flatnonzero((p.size == k) & (p.size > 0)))
            fixed.append(replace(p, release=k, rows=rows, fixed=np.arange(k+1)))
        at = np.flatnonzero(rec == r)
        status, edits = study.exact(fixed, gain, risk, moments["easy"], before[at].sum(),
                                    moments["ref_easy"][at].sum(), 0)
        cap[status] += 1
        if status == "qualified":
            # Replace only recipients, keep the same number of edits at every query.
            witness[at] = cv[at]
            witness[edits] = path[edits]
            actual = np.bincount(d["query"][at[(learned[at] != cv[at]).any((1, 2))]])
            changed = np.bincount(d["query"][edits])
            np.testing.assert_array_equal(actual, changed)
            stat = v.parent.score(dict(y=y[at], valid=valid[at], rec=rec[at]), cv[at], ols[at], witness[at], cut)
            if stat["dual_positive"] != 1:
                raise ValueError("Matched-clock witness does not pass the original rule")
    learned_metrics = v.parent.score(d, cv, ols, learned, cut)
    if cap["qualified"]+cap["unresolved"] < learned_metrics["dual_positive"]:
        raise ValueError("The exact menu must contain the causal output")
    return dict(capacity=cap, learned={f: learned_metrics[f] for f in v.prior.FIELDS})


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    k, outer, old, source, *_ = study.register()
    path, heads = OUT/"analysis.json", OUT/"head_freeze.json"
    a, h = run.read(path), run.read(heads)
    if (a["registration"] != k or h["registration"] != k or a["heads"] != run.sha(heads)
            or not a["original_2pct"] or a["confirmation_access"]):
        raise ValueError("Require intact frozen slate results")
    f = OUT/"clock_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(path),
        control="label-aware subset assignment at every learned realized spend count",
        preserves_recording_query_counts=True, original_2pct=True, confirmation_access=False)))
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, starts) in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = study.hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            release = study.two.fit.load(outer[f"{s}_{seed}"], lambda: study.hard.model.network(width), study.hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            p = next(z for _, seq in study.packets(d, *bank[:2], path, starts, release) for z in seq)
            n = m.actions(p, (0, 0))[1].shape[1]
            net = study.two.fit.load(h["heads"][f"{s}_{seed}_full"], lambda: v.model.network(n), study.STEPS)
            r = assess(d, bank, path, starts, release, net, cut)
            cell = next(z for z in a["by_site"][s] if z["seed"] == seed)
            for field in v.prior.FIELDS:
                if r["learned"][field] != cell["metrics"]["adaptive_queue"][field]:
                    raise ValueError("Frozen learned clock did not reproduce")
            result[s].append(dict(seed=seed, **r))
        print(s, "matched-clock ceiling complete", flush=True)
    total = sum((r["capacity"] for group in result.values() for r in group), Counter())
    run.once(OUT/"clock_capacity.json", run.stable(dict(registration=run.sha(f), by_site=result,
        summary=total, original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
