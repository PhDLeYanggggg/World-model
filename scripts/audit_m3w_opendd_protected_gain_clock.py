"""Bound delayed gain with the exact same support action and peak radius."""

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
from scripts import run_m3w_opendd_protected_gain as main_run
from src.world_model.m3w_opendd_record_support import support_targets

v, two, run, hard, prior, record = main_run.v, main_run.two, main_run.run, main_run.hard, main_run.prior, main_run.record
OUT = main_run.OUT
shape, stop, base = main_run.rank.shape, record.stop, record.base
KEYS = ("current_protected", "delay_protected", "delay_reference_preserved",
        "current_any", "delay_any")


@torch.inference_mode()
def evaluate(d, bank, path, release, gain, analytic, cuts, ref):
    cv, ols = bank[:2]
    roster, small, large, _, _ = record.menus(d, cv, path, release, gain)
    y, valid, q, rec = (d[k] for k in ("y", "valid", "query", "rec"))
    count = Counter({k: 0 for k in KEYS})
    for r, big_row, support, _, _ in roster:
        at = np.flatnonzero(rec == r)
        cv0, ols0, y0, valid0 = cv[at], ols[at], y[at], valid[at]
        d0 = dict(y=y0, valid=valid0, rec=rec[at])
        p = cv0.copy()
        s = int(np.searchsorted(at, support)) if support is not None else None
        if s is not None:
            p[s] = small[support]
        if big_row is None:
            passed = v.parent.score(d0, cv0, ols0, p, cuts)["dual_positive"]
            count.update({k: passed for k in KEYS})
            continue
        g = int(np.searchsorted(at, big_row))
        radius = shape.peak((large[big_row]-cv[big_row])[None])[0]
        a = record.contributions(cv0, ols0, p, y0, valid0, cuts[0])
        full_easy, offset = a[2].sum(), a[4].sum()
        fixed = tuple(x[s] for x in a[:4]) if s is not None else None
        support_easy = bool(s is not None and a[2][s] > 0.)
        original = bool(a[2][g] > 0. or support_easy)
        time = q[at]
        current = time == time[g]
        later = time >= time[g]
        observed = (np.arange(len(at)) == g) | ((time >= time[s]) if s is not None else False)
        reserve_row = np.arange(len(at)) != s
        hit = {k: False for k in KEYS}
        for raw in (large[at], ols0, analytic[at]):
            delta = raw-cv0
            size = shape.peak(delta)
            legal = (size >= radius) & (radius > 0.) & reserve_row & later
            candidate = cv0.copy()
            candidate[legal] += delta[legal]*(radius/size[legal])[:, None, None]
            args = record.contributions(cv0, ols0, candidate, y0, valid0, cuts[0])
            _, feasible = support_targets(*args[:4], full_easy, fixed, offset)
            feasible &= legal
            no_ref_loss = (not original) | (args[2] > 0.) | support_easy
            masks = (current & observed, later & observed, later & observed & no_ref_loss, current, later)
            for k, m in zip(KEYS, masks, strict=True):
                ok = feasible & m
                if hit[k] or not ok.any():
                    continue
                j = np.flatnonzero(ok)[0]
                f = p.copy()
                f[j] = candidate[j]
                np.testing.assert_allclose(shape.peak((f[j]-cv0[j])[None])[0], radius, atol=1e-12, rtol=1e-10)
                score = v.parent.score(d0, cv0, ols0, f, cuts)
                if score["dual_positive"] != 1 or score["edits"] != 1+int(s is not None):
                    raise ValueError("Delayed witness did not match the exact risk scorer")
                hit[k] = True
        if hit["current_protected"] > hit["delay_protected"] or hit["delay_protected"] > hit["delay_any"]:
            raise ValueError("Expanded clock lost feasible actions")
        count.update(hit)
    if (count["current_protected"] < ref["metrics"]["oracle_protected"]["dual_positive"]
            or count["current_any"] < ref["metrics"]["oracle_gain"]["dual_positive"]):
        raise ValueError("All-row menu does not cover the prior bounded shortlist")
    return count


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, gains, *_ = main_run.register()
    p, h = (OUT/f"{k}.json" for k in ("analysis", "head_freeze"))
    a = run.read(p)
    if (a["registration"] != key or a["heads"] != run.sha(h)
            or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require the frozen protected-gain readout")
    f = OUT/"clock_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p), run.artifact(h)],
        original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        matching="exact support trajectory and original gain radius/count, gain query may only move later",
        menu="all legal rows; no amplification; before support the protected gain row stays unchanged")))
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    result = {}
    for s, d in sites.items():
        bank = v.parent.prior.analytic(d)
        cuts = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        analytic = shape.analytic(d["h"])[2]
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            name = f"{s}_{seed}"
            release = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            cv = bank[0]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
            gain = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            ref = next(x for x in a["by_site"][s] if x["seed"] == seed)
            result[s].append(dict(seed=seed, counts=evaluate(d, bank, path, release, gain, analytic, cuts, ref)))
        print(s, "protected gain clock audited", flush=True)
    total = sum((r["counts"] for group in result.values() for r in group), Counter())
    run.once(OUT/"clock_capacity.json", run.stable(dict(registration=run.sha(f), summary=total, by_site=result,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))
    print(run.stable(total), flush=True)


if __name__ == "__main__":
    main()
