"""Attribute release-to-offer expiry and bound the eight-query action menu."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_gain_wait as study
from src.world_model.m3w_opendd_record_support import support_targets

run, record, v, two, hard, prior = study.run, study.record, study.v, study.two, study.hard, study.prior
OUT = study.OUT
BOUNDS = ("after_support_first", "first_neural_query", "any_neural", "first_any_query", "any_direction")


@torch.inference_mode()
def audit(d, cv, ols, path, pack, cuts, ref, broad):
    roster, _, _, original, support, waited = pack
    rec, q, y, valid = (d[k] for k in ("rec", "query", "y", "valid"))
    support_edit = (support != cv).any((1, 2))
    raw = cv+study.clock.base.model.GAIN_SCALE*(path-cv)
    analytic = study.shape.analytic(d["h"])[2]
    truth = study.labels(d, cv, ols, cuts, pack)[1]
    count = Counter({k: 0 for k in (*BOUNDS, "expired_no_support", "expired_no_later_query",
                                    "expired_support_after_deadline", "expired_no_radius_match",
                                    "expired_but_other_direction_available")})
    for (r, first, offer, _, seen), label in zip(roster, truth, strict=True):
        at = np.flatnonzero(rec == r)
        x, o, y0, ok = cv[at], ols[at], y[at], valid[at]
        s = np.flatnonzero(support_edit[at])
        if len(s) > 1:
            raise ValueError("More than one frozen support edit")
        start = int(np.searchsorted(at, first))
        time = np.searchsorted(np.unique(q[at]), q[at])
        t = time[start]
        end = t+study.model.DEADLINE
        permitted = (time > t) & (time <= end)
        legal = permitted & (time >= time[s[0]]) if len(s) else np.zeros(len(at), dtype=bool)
        legal[s] = False
        if offer is None:
            reason = ("no_support" if not len(s) else
                      "no_later_query" if not permitted.any() else
                      "support_after_deadline" if time[s[0]] > end else "no_radius_match")
            count["expired_"+reason] += 1
        if seen and label["new_joint"]:
            count["after_support_first"] += 1

        support_terms = record.contributions(x, o, support[at], y0, ok, cuts[0])
        selected = tuple(a[s[0]] for a in support_terms[:4]) if len(s) else None
        original_defined = bool(support_terms[2][start] > 0. or (len(s) and support_terms[2][s[0]] > 0.))
        orig = record.contributions(x, o, original[at], y0, ok, cuts[0])
        r0 = orig[2].sum()
        offset = orig[4].sum()
        a = orig[0][start:start+1]
        old_joint = support_targets(a, orig[1][start:start+1], orig[2][start:start+1],
                                    orig[3][start:start+1], r0, selected, offset)[1][0]
        if old_joint and label["new_joint"]:
            raise ValueError("Unexpected oracle transition")
        radius = study.shape.peak((original[first]-cv[first])[None])[0]
        masks, good, proposals = [], [], []
        for p in (raw[at], o, analytic[at]):
            delta = p-x
            size = study.shape.peak(delta)
            match = legal & (size >= radius)
            pred = x.copy()
            pred[match] += delta[match]*(radius/size[match])[:, None, None]
            g, excess, easy, base, _ = record.contributions(x, o, pred, y0, ok, cuts[0])
            joint = support_targets(g, excess, easy, base, r0, selected, offset)[1]
            preserve = (not original_defined) | (easy > 0.) | (bool(len(s)) and support_terms[2][s[0]] > 0.)
            masks.append(match)
            good.append(joint & preserve & match)
            proposals.append(pred)
        any_offer = np.logical_or.reduce(masks)
        first_query = lambda m: (time == time[m].min()) if m.any() else np.zeros(len(at), dtype=bool)
        if bool(masks[0].any()) != (offer is not None):
            raise ValueError("Causal first-offer availability disagrees with exact menu")
        if offer is not None:
            j = int(np.searchsorted(at, offer))
            np.testing.assert_allclose(waited[offer], proposals[0][j], rtol=1e-10, atol=1e-12)
            if not first_query(masks[0])[j] or (label["new_joint"] and not good[0][j]):
                raise ValueError("First-offer witness inconsistent")
        if offer is None and any_offer.any():
            count["expired_but_other_direction_available"] += 1
        if old_joint:
            continue
        for k, opts, use in (("first_neural_query", [0], first_query(masks[0])),
                             ("any_neural", [0], legal),
                             ("first_any_query", range(3), first_query(any_offer)),
                             ("any_direction", range(3), legal)):
            for i in opts:
                hit = np.flatnonzero(good[i] & use)
                if not len(hit):
                    continue
                j = hit[0]
                p = support[at].copy()
                p[j] = proposals[i][j]
                score = v.parent.score(dict(y=y0, valid=ok, rec=rec[at]), x, o, p, cuts)
                if score["dual_positive"] != 1 or score["edits"] != 1+len(s):
                    raise ValueError("Bound witness failed exact original scoring")
                count[k] += 1
                break

    original_pass = ref["metrics"]["original"]["dual_positive"]
    count.update({k: original_pass for k in BOUNDS})
    count["original"] = original_pass
    count["causal_menu_oracle"] = ref["metrics"]["binary_oracle"]["dual_positive"]
    if sum(n for k, n in count.items() if k.startswith("expired_") and k != "expired_but_other_direction_available") != ref["diagnostics"]["expired"]:
        raise ValueError("Expiry attribution failed to reconcile")
    if not (count["after_support_first"] <= count["causal_menu_oracle"] <= count["first_neural_query"]
            <= count["any_neural"] <= count["any_direction"] <= broad["delay_reference_preserved"]
            and count["first_any_query"] <= count["any_direction"]):
        raise ValueError("Inconsistent capacity between restricted menus")
    return count


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, gains, *_ = study.register()
    a = study.timing.legacy.read(OUT.name)
    cap = study.timing.read_protected_clock()
    if a["registration"] != key:
        raise ValueError("Require the frozen gain-wait run")
    f = OUT/"menu_registration.json"
    parents = [OUT/"analysis.json", OUT/"head_freeze.json", study.parent.OUT/"clock_capacity.json"]
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p) for p in parents],
        original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        expiry_attribution="no support, no later query, support after deadline, or no radius match",
        bound="keep original or replace gain once after support; exact radius and support; preserve easy reference")))
    sites = {s: study.study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    result = {}
    for s, d in sites.items():
        cv, ols = v.parent.prior.analytic(d)[:2]
        cuts = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            name = f"{s}_{seed}"
            release = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
            first = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            lookup = lambda doc: next(z for z in doc["by_site"][s] if z["seed"] == seed)
            pack = study.prepare(d, cv, path, release, first)
            result[s].append(dict(seed=seed, counts=audit(d, cv, ols, path, pack, cuts, lookup(a), lookup(cap)["counts"])))
        print(s, "gain-wait capacity audited", flush=True)
    total = sum((z["counts"] for rows in result.values() for z in rows), Counter())
    run.once(OUT/"menu_capacity.json", run.stable(dict(registration=run.sha(f), summary=total, by_site=result,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))
    print(run.stable(total), flush=True)


if __name__ == "__main__":
    main()
