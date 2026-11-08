"""Check whether adding the frozen support edit moves recording risk monotonically."""

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
from src.evaluation.m3w_selection_forecast_readout import policy

run, v, two, hard, prior = record.run, record.v, record.two, record.hard, record.prior
OUT = record.OUT.parent/"opendd_support_monotonicity_v1"
RISKS = ("completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation")


def status(row):
    if any(row[k] is None for k in RISKS):
        return "undefined"
    return "pass" if row["supported"] else "over_limit"


def evaluate(d, cv, path, release, head, cut, ref):
    roster, small, large, _, _ = record.menus(d, cv, path, release, head)
    gain, both = cv.copy(), cv.copy()
    active = set()
    for r, first, anchor, _, _ in roster:
        if first is not None:
            gain[first] = both[first] = large[first]
        if anchor is not None:
            both[anchor] = small[anchor]
            active.add(str(r))
    def score(p):
        return policy(p, cv, cv, d["y"], d["valid"], d["rec"],
                      easy_cut=cut[0], hard_cut=cut[1])["recordings"]

    a, b = score(gain), score(both)
    for key, z in (("gain", a), ("two_roles", b)):
        fields = dict(risk_pass=sum(row["supported"] for row in z.values()),
                      edits=sum(row["selected"] for row in z.values()),
                      undefined=sum(row[RISKS[1]] is None for row in z.values()))
        if any(ref["metrics"][key][k] != n for k, n in fields.items()):
            raise ValueError("Risk audit did not recover the frozen reference")
    counts, moves = Counter(), {k: Counter() for k in RISKS}
    for r in active:
        counts[f"{status(a[r])}_to_{status(b[r])}"] += 1
        for k in RISKS:
            u, w = a[r][k], b[r][k]
            if u is None or w is None:
                moves[k]["not_both_estimable"] += 1
                continue
            moves[k]["increase" if w-u > 1e-12 else "decrease" if u-w > 1e-12 else "unchanged"] += 1
            if u <= .02 < w:
                moves[k]["cross_up"] += 1
            if w <= .02 < u:
                moves[k]["cross_down"] += 1
    return dict(active=len(active), transitions=counts, numeric=moves)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    key, outer, old, source, gain, ref, *_ = record.admit.register()
    parent = record.admit.OUT/"analysis.json"
    a = run.read(parent)
    if a["registration"] != key or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require the frozen support readout")
    OUT.mkdir(parents=True, exist_ok=True)
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(parent),
        recipient_reader=run.artifact(Path(record.__file__)), no_training=True,
        intervention="same gain actions; compare adding the frozen easy support action in each of its 345 active views",
        undefined_kept=True, original_2pct=True, confirmation_access=False)))
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    rows = {}
    for s, d in sites.items():
        cv = v.parent.prior.analytic(d)[0]
        cuts = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        rows[s] = []
        for seed in two.study.prior.SEEDS:
            name = f"{s}_{seed}"
            release = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            lookup = lambda b: next(z for z in b["by_site"][s] if z["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as p:
                path = p["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
            head = two.fit.load(gain[name], lambda: prior.model.network(n), prior.STEPS)
            rows[s].append(dict(seed=seed, **evaluate(d, cv, path, release, head, cuts, lookup(ref))))
        print(s, "monotonicity audited", flush=True)
    flat = [r for group in rows.values() for r in group]
    summary = dict(active=sum(r["active"] for r in flat), transitions=sum((r["transitions"] for r in flat), Counter()),
        numeric={k: sum((r["numeric"][k] for r in flat), Counter()) for k in RISKS})
    if summary["active"] != a["summary"]["detail"]["easy"]["support"]:
        raise ValueError("Changed support coverage")
    run.once(OUT/"analysis.json", run.stable(dict(registration=run.sha(reg), summary=summary, by_site=rows,
        original_2pct=True, confirmation_access=False, submission_ready=False)))
    print(run.stable(summary), flush=True)


if __name__ == "__main__":
    main()
