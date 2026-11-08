"""Factor timing and recipient-menu limits at the same second-slot budget."""

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
from scripts import run_m3w_opendd_support_stop as stop

v, run, two, hard, prior, base = stop.v, stop.run, stop.two, stop.hard, stop.prior, stop.base
OUT = stop.OUT
CLOCKS = ("fixed", "plus_two", "after_release", "any_query")
MENUS = ("top_easy", "all_predicted_easy", "all_legal")


@torch.inference_mode()
def evaluate(d, bank, path, release, gain_head, cuts, earlier, ceiling, base_cell):
    cv, ols = bank[:2]
    p, _, _, easy, blocks = stop.inputs(d, cv, path, release)
    large = cv+base.model.GAIN_SCALE*(path-cv)
    f, scale = prior.feature(d, cv, large)
    gain = gain_head(torch.from_numpy(f)).numpy()[:, 0]*scale
    y, valid, rec = d["y"], d["valid"], d["rec"]
    counts = {k: Counter() for k in CLOCKS}
    for r, rows in blocks.items():
        first, second, _ = base.model.choose((ix, gain[ix], easy[ix]) for ix in rows)
        at = np.flatnonzero(rec == r)
        cv0, ols0, y0, mask = cv[at], ols[at], y[at], valid[at]
        floor = cv0.copy()
        if first is not None:
            floor[np.searchsorted(at, first)] = large[first]
        d0 = dict(y=y0, valid=mask, rec=rec[at])
        if second is None:
            j = v.parent.score(d0, cv0, ols0, floor, cuts)["dual_positive"]
            for k in CLOCKS:
                counts[k].update({a: j for a in MENUS})
            continue
        top = stop.options(rows, easy, first)
        i = top.index(second)
        windows = (range(i, i+1), range(i, min(len(rows), i+stop.model.DEADLINE+1)),
                   range(i, len(rows)), range(len(rows)))
        default = floor.copy()
        default[np.searchsorted(at, second)] = p[second]
        verified_default = v.parent.score(d0, cv0, ols0, default, cuts)["dual_positive"]
        for clock, t in zip(CLOCKS, windows, strict=True):
            pool = np.concatenate([rows[j] for j in t])
            chosen = np.asarray([top[j] for j in t if top[j] is not None], dtype=int)
            for menu, ix in zip(MENUS, (chosen, pool[easy[pool] >= base.model.EASY_THRESHOLD], pool), strict=True):
                local = np.searchsorted(at, ix)
                candidate = cv0.copy()
                candidate[local] = p[ix]
                winner = base.best_second(cv0, ols0, candidate, y0, mask, floor, cuts[0])
                if winner is not None:
                    pred = floor.copy()
                    pred[winner] = candidate[winner]
                    score = v.parent.score(d0, cv0, ols0, pred, cuts)
                    if score["dual_positive"] != 1 or score["edits"] != int(first is not None)+1:
                        raise ValueError("Menu witness failed original risk or count checks")
                counts[clock][menu] += int(winner is not None or verified_default)
    known = (counts["fixed"]["top_easy"] == earlier["metrics"]["immediate"]["dual_positive"]
             and counts["plus_two"]["top_easy"] == earlier["metrics"]["bounded_oracle"]["dual_positive"]
             and counts["fixed"]["all_legal"] == base_cell["metrics"]["oracle_same_query"]["dual_positive"]
             and counts["any_query"]["all_legal"] == ceiling["any_query_joint"])
    if not known:
        raise ValueError("One of the pre-existing capacity controls changed")
    for i, k in enumerate(CLOCKS):
        if any(counts[k][a] > counts[k][b] for a, b in zip(MENUS[:-1], MENUS[1:], strict=True)):
            raise ValueError("Expanded recipient menu lost a witness")
        if i and any(counts[CLOCKS[i-1]][a] > counts[k][a] for a in MENUS):
            raise ValueError("Expanded timing window lost a witness")
    return counts


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, gain_heads, ref, _, _, cap = stop.register()
    p = OUT/"analysis.json"
    a = run.read(p)
    if a["registration"] != key or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require verified support-stop evidence")
    f = OUT/"menu_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(p),
        original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        clocks=CLOCKS, menus=MENUS, no_new_training=True,
        matching="same first edit, second-slot recordings, second count and amplitude")))
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, _) in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            name = f"{s}_{seed}"
            rel = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            gain = two.fit.load(gain_heads[name], lambda: prior.model.network(n), prior.STEPS)
            old_row = lambda z: next(q for q in z["by_site"][s] if q["seed"] == seed)
            val = evaluate(d, bank, path, rel, gain, cut, old_row(a), old_row(cap)["counts"], old_row(ref))
            result[s].append(dict(seed=seed, counts=val))
        print(s, "support timing-by-menu bound complete", flush=True)
    summary = {k: sum((r["counts"][k] for z in result.values() for r in z), Counter()) for k in CLOCKS}
    run.once(OUT/"menu_capacity.json", run.stable(dict(registration=run.sha(f), summary=summary,
        by_site=result, original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        submission_ready=False)))


if __name__ == "__main__":
    main()
