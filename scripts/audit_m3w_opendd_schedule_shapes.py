"""Test expanded same-peak forecast directions with a flexible two-role clock."""

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
from scripts import audit_m3w_opendd_schedule_capacity as capacity
from src.world_model import m3w_opendd_reserve_shape as shape

schedule, run, v = capacity.schedule, capacity.run, capacity.v
OUT = schedule.OUT


def matched_direction(cv, neural, other):
    delta = other-cv
    r, limit = shape.peak(neural-cv), shape.peak(delta)
    legal = (r > 0.) & (limit >= r)
    p = cv.copy()
    p[legal] = cv[legal]+delta[legal]*(r[legal]/limit[legal])[:, None, None]
    np.testing.assert_allclose(shape.peak(p[legal]-cv[legal]), r[legal], rtol=1e-10, atol=1e-12)
    return p


def find_action(big, small, ref, offset):
    def subset(z, keep):
        return tuple(x[keep] for x in z)

    for role, (parts, _, _) in enumerate((big, small)):
        _, ok = schedule.model.support_targets(*parts, ref, ols_offset=offset)
        if ok.any():
            j = int(np.flatnonzero(ok)[0])
            return (j, None) if role == 0 else (None, j)
    bp, b_rows, _ = big
    sp, s_rows, _ = small
    for j, row in enumerate(b_rows):
        keep = s_rows != row
        _, ok = schedule.model.support_targets(*subset(sp, keep), ref,
            first=tuple(z[j] for z in bp), ols_offset=offset)
        if ok.any():
            return j, int(np.flatnonzero(keep)[np.flatnonzero(ok)[0]])
    return None


def evaluate(d, baseline, path, analytic, cuts, previous):
    cv, ols = baseline[:2]
    y, valid, rec = d["y"], d["valid"], d["rec"]
    opts = []
    for alpha in (capacity.base.model.GAIN_SCALE, capacity.base.model.SUPPORT_SCALE):
        neural = cv+alpha*(path-cv)
        p = (neural, matched_direction(cv, neural, ols), matched_direction(cv, neural, analytic))
        opts.append([(x, capacity.record.contributions(cv, ols, x, y, valid, cuts[0])) for x in p])
    total = Counter(neural=0, ols_menu=0, prefix_menu=0, all_menu=0)
    for r in np.unique(rec):
        ix = np.flatnonzero(rec == r)
        d0 = dict(y=y[ix], valid=valid[ix], rec=rec[ix])
        ref, off = [x[ix].sum() for x in opts[0][0][1][2:5:2]]

        def role(z, families):
            blocks = []
            for k in families:
                p, c = z[k]
                j = ix[(p[ix] != cv[ix]).any((1, 2))]
                blocks.append((tuple(x[j] for x in c[:4]), j, p[j]))
            return (tuple(np.concatenate(a) for a in zip(*(x[0] for x in blocks), strict=True)),
                    np.concatenate([x[1] for x in blocks]), np.concatenate([x[2] for x in blocks]))

        hits = []
        for label, kinds in (("neural", (0,)), ("ols_menu", (0, 1)), ("prefix_menu", (0, 2)),
                             ("all_menu", (0, 1, 2))):
            parts = [role(z, kinds) for z in opts]
            choice = find_action(*parts, ref, off) if ref > 0. else None
            total[label] += int(choice is not None)
            hits.append(choice is not None)
            if choice is None:
                continue
            p = cv[ix].copy()
            rows = set()
            for j, (_, row, paths) in zip(choice, parts, strict=True):
                if j is not None:
                    rows.add(int(row[j]))
                    p[np.searchsorted(ix, row[j])] = paths[j]
            n = sum(j is not None for j in choice)
            a = v.parent.score(d0, cv[ix], ols[ix], p, cuts)
            if len(rows) != n or a["edits"] != n or a["dual_positive"] != 1:
                raise ValueError("Expanded shape witness changed budget or scoring")
        if any(hits[0] > i or i > hits[3] for i in hits[1:3]):
            raise ValueError("Expanding a forecast menu cannot remove feasibility")
    if total["neural"] != previous["any_up_to_two"]:
        raise ValueError("Original two-role shape bound did not replay")
    return total


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, _, old, source, *_ = schedule.register()
    parents = [OUT/f"{k}.json" for k in ("analysis", "capacity_breakdown", "capacity_registration")]
    a, b, c = map(run.read, parents)
    if (a["registration"] != key or b["registration"] != run.sha(parents[2])
            or c["parent"]["sha256"] != run.sha(parents[0])
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, b, c))):
        raise ValueError("Require verified schedule feasibility audit")
    f = OUT/"shape_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p) for p in parents],
        original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        matching="same neural per-row per-role peak; no amplification or extra editable rows; at most one action per role",
        menu="neural, OLS4 direction, observed-prefix analytic direction; any clock")))
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    result = {}
    for s, d in sites.items():
        bank = v.parent.prior.analytic(d)
        cuts = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        _, _, analytic, _ = shape.analytic(d["h"])
        result[s] = []
        for row in old["by_site"][s]:
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                p = z["comoving_ego"]
            ref = next(z["capacity"] for z in a["by_site"][s] if z["seed"] == row["seed"])
            result[s].append(dict(seed=row["seed"], counts=evaluate(d, bank, p, analytic, cuts, ref)))
        print(s, "same-peak shape expansion audited", flush=True)
    summary = sum((r["counts"] for group in result.values() for r in group), Counter())
    run.once(OUT/"shape_capacity.json", run.stable(dict(registration=run.sha(f), summary=summary, by_site=result,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))
    print(run.stable(summary), flush=True)


if __name__ == "__main__":
    main()
