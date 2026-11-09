"""Retrieve exact-slot OpenDD slates by both feasibility and relative gain."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

from scipy.stats import rankdata

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import report_m3w_conditional_pair_gain as report

old = report.run
pair, np, torch, study, base = old.pair, old.np, old.torch, old.study, old.base
OUT, LOCAL = old.OUT.parent/"opendd_joint_retrieval_v1", old.LOCAL.parent/"opendd_joint_retrieval_v1"
METHODS = ("feasibility8", "joint8")
WIDTH, KEEP = old.SHORTLIST, old.SHORTLIST//2


def beat(**info):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), **info))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    old_key, cohorts, _ = old.register()
    p = old.OUT/"summary.json"
    r = study.read(p)
    f = study.read(study.checked(r["freeze"]))
    if r["registration"] != old_key or f["registration"] != old_key or f["held_outcomes_used"] or r["reserved_outcomes_opened"]:
        raise ValueError("Expected the frozen conditional selector")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(parent=study.artifact(p), parent_freeze=r["freeze"],
        previous_shortlist_audit=study.artifact(old.OUT/"shortlist_audit.json"),
        code=[study.artifact(Path(x)) for x in (__file__, old.__file__, report.__file__)],
        arms=METHODS, retain=KEEP, final_shortlist=WIDTH,
        retrieval="four highest feasibility then unused highest equal-weight average ranks of feasibility and full-grid relative gain",
        final_ranking="both choose highest positive predicted conservative pair gain; otherwise top feasibility",
        match="same frozen weights, reference, legal assignment menu and per-query radii; more retrieval computation",
        calibration="unchanged grid, all original nonempty 2 percent tests for both arms",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, new_training=False))
    return base.sha(reg), cohorts, f


def retrieve(feasibility, gain):
    if len(feasibility) != len(gain) or not len(gain) or not np.isfinite(feasibility).all() or not np.isfinite(gain).all():
        raise ValueError("Require finite full-menu scores for both objectives")
    high = np.argsort(-feasibility, kind="stable")
    fused = rankdata(feasibility)+rankdata(gain)
    order = np.argsort(-fused, kind="stable")
    initial = high[:KEEP]
    joint = np.r_[initial, order[~np.isin(order, initial)]][:WIDTH]
    return high[:WIDTH], joint


def scored_gain(net, m, paths, a):
    x, xr, b, _, _ = old.relative.features(m["h"], m["c"], m["ols"], paths, a)
    v = net(*(torch.from_numpy(z) for z in (x, xr, b))).numpy()
    return v[:, 0]-v[:, 1]


@torch.inference_mode()
def infer(gain, feasible, inp, beta):
    cv, q = inp["cv"], inp["q"]
    p = {k: cv.copy() for k in ("forced_add", *METHODS)}
    scores = {k: np.full(len(cv), -1.) for k in METHODS}
    short = {k: np.full((q.max()+1, WIDTH), -1, dtype=int) for k in METHODS}
    counts = Counter()
    for ix, m in pair.menus(inp):
        a = old.reference(m, beta)
        p["forced_add"][ix] = a
        f = []
        for k in range(0, len(m["opts"]), 256):
            x, g = pair.encode(m, m["opts"][k:k+256])
            f.append(feasible(torch.from_numpy(x), torch.from_numpy(g)).sigmoid().numpy()[:, 1, 0])
        f = np.concatenate(f)
        u = np.concatenate([scored_gain(gain, m, pair.paths_for(m, m["opts"][k:k+128]), a)
                            for k in range(0, len(m["opts"]), 128)])
        choices = retrieve(f, u)
        for method, shortlist in zip(METHODS, choices, strict=True):
            candidates = pair.paths_for(m, m["opts"][shortlist])
            s = scored_gain(gain, m, candidates, a)
            best = int(s.argmax()) if s.max() > 0. else 0
            p[method][ix] = candidates[best]
            scores[method][ix] = f[shortlist[best]]
            short[method][q[ix.start], :len(shortlist)] = shortlist
        counts["queries"] += 1
        counts["candidate_scores"] += len(u)
        counts["changed_predictions"] += int(not np.allclose(p["feasibility8"][ix], p["joint8"][ix], rtol=1e-9, atol=1e-9))
        counts["changed_shortlist"] += int(not np.array_equal(np.sort(choices[0]), np.sort(choices[1])))
        for arm in p:
            r = pair.run.turn.turn.peak(p[arm][ix]-cv[ix])
            np.testing.assert_allclose(np.sort(r[r > 0.]), np.sort(m["radius"]), atol=1e-8, rtol=1e-8)
    return dict(q=q, cv=cv, ols4=inp["ols"], **p, **{f"score_{k}": v for k, v in scores.items()},
                **{f"short_{k}": v for k, v in short.items()}), dict(counts)


def load_models(item, site, seed, train, cal):
    fit = item["fit"]
    checkpoint = torch.load(study.checked(fit["checkpoint"]), weights_only=True, map_location="cpu")
    refit = checkpoint["identity"]
    if (refit != fit["identity"] or checkpoint["step"] != old.STEPS or refit["held"] != site or refit["cal"] != cal
            or refit["seed"] != seed or not set(refit["donors"]) <= set(train)):
        raise ValueError("Wrong relative gain fold")
    if not set(item["reference_sites"]) <= set(train):
        raise ValueError("Additive reference used evaluation or calibration labels")
    context = torch.load(study.checked(item["feasibility"]), weights_only=True, map_location="cpu")
    ident = context["identity"]
    if ident["held"] != site or ident["cal"] != cal or ident["seed"] != seed or not set(ident["donors"]) <= set(train):
        raise ValueError("Wrong feasibility fold")
    g, f = old.relative.RelativeGain().eval(), pair.model.PairAdmission().eval()
    g.load_state_dict(checkpoint["net"])
    f.load_state_dict(context["net"])
    return g, f


def freeze(key, cohorts, parent):
    data = pair.run.turn.exp.loader(cohorts)
    inputs = {s: pair.causal(cohorts[s]) for s in old.SITES}
    result = {}
    for site in old.SITES:
        cal, train = pair.roles(site)
        for seed in old.SEEDS:
            name = f"{site}_{seed}"
            path = LOCAL/f"{name}.json"
            item = parent["folds"][name]
            if not path.exists():
                gain, feasibility = load_models(item, site, seed, train, cal)
                roles = {}
                for role, s in (("cal", cal), ("held", site)):
                    p, counts = infer(gain, feasibility, inputs[s], np.asarray(item["reference"]))
                    prev = old.rank.checked_npz(item[role]["forecast"])
                    for new, before in (("forced_add", "forced_add"), ("feasibility8", "pair_gain"),
                                        ("score_feasibility8", "score_pair_gain")):
                        np.testing.assert_array_equal(p[new], prev[before])
                    thresholds = {}
                    if role == "cal":
                        for k in METHODS:
                            thresholds[k] = pair.choose_threshold(old.as_policy(p, k), data[s], item["cut"])[0]
                        if thresholds["feasibility8"] != item["cal"]["thresholds"]["pair_gain"]:
                            raise ValueError("Control calibration did not replay")
                    artifact = pair.run.turn.exp.previous.save_bank(LOCAL/f"{name}_{role}.npz", p)
                    roles[role] = dict(forecast=artifact, counts=counts, thresholds=thresholds)
                base.immutable(path, dict(registration=key, cal_site=cal, held_site=site, seed=seed,
                    cut=item["cut"], parent=item["fit"]["checkpoint"], **roles))
            saved = study.read(path)
            if (saved["registration"] != key or saved["cal_site"] != cal or saved["held_site"] != site
                    or saved["parent"] != item["fit"]["checkpoint"] or saved["seed"] != seed):
                raise ValueError("Fold or source roles changed")
            for role in ("cal", "held"):
                study.checked(saved[role]["forecast"])
            result[name] = saved
            beat(phase="freeze", fold=name, done=len(result), total=len(old.SITES)*len(old.SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=result, held_targets_used=False))


def audit(inputs, d, p, cut):
    stats = Counter()
    for ix, m in pair.menus(inputs):
        q = p["q"][ix.start]
        a, y, v = p["forced_add"][ix], d["y"][ix], d["v"][ix]
        qualified = np.concatenate([old.target.targets(m, pair.paths_for(m, m["opts"][k:k+128]), a, y, v, cut)[1]
                                    for k in range(0, len(m["opts"]), 128)])
        stats.update(queries=1, feasible_pair=int(qualified.any()))
        captured = {}
        for method in METHODS:
            take = p[f"short_{method}"][q]
            take = take[take >= 0]
            if len(take) != min(WIDTH, len(qualified)) or len(set(take)) != len(take):
                raise ValueError("Changed fixed shortlist budget")
            captured[method] = bool(qualified[take].any())
            picked = bool(old.target.targets(m, p[method][ix][None], a, y, v, cut)[1][0])
            if picked and not captured[method]:
                raise ValueError("Selected an action outside the shortlist")
            stats[f"retained_{method}"] += int(captured[method])
            stats[f"selected_{method}"] += int(picked)
        stats["newly_retrieved"] += int(captured["joint8"] and not captured["feasibility8"])
        stats["lost"] += int(captured["feasibility8"] and not captured["joint8"])
    return dict(stats)


def contrast(p, d):
    rows = []
    for rec in np.unique(d["rec"]):
        at = d["rec"] == rec
        pred = dict(cv=p["cv"][at], feasibility=p["feasibility8"][at], pair_gain=p["joint8"][at])
        diff = report.direct(pred, {k: d[k][at] for k in ("q", "y", "v")})
        if diff is not None:
            rows.append(dict(record=str(rec), **diff))
    return rows


def score(key, cohorts):
    path = OUT/"freeze.json"
    frozen = study.read(path)
    if (frozen["registration"] != key or frozen["held_targets_used"]
            or set(frozen["folds"]) != {f"{s}_{t}" for s in old.SITES for t in old.SEEDS}):
        raise ValueError("Require all held forecasts frozen")
    data = pair.run.turn.exp.loader(cohorts)
    inp = {s: pair.causal(cohorts[s]) for s in old.SITES}
    rows, sites = ({m: [] for m in METHODS} for _ in range(2))
    captures, diffs, thresholds = {}, {}, {}
    counts = Counter()
    for site in old.SITES:
        local, change = {m: [] for m in METHODS}, []
        for seed in old.SEEDS:
            name = f"{site}_{seed}"
            item = frozen["folds"][name]
            cut = item["cut"]
            captures[name], thresholds[name] = {}, {}
            for role, s in (("cal", item["cal_site"]), ("held", site)):
                d = data[s]
                p = old.rank.checked_npz(item[role]["forecast"])
                captures[name][role] = audit(inp[s], d, p, cut[0])
                for m in METHODS:
                    policy = old.as_policy(p, m)
                    t = item["cal"]["thresholds"][m]
                    if role == "cal":
                        thresholds[name][m] = dict(grid=t, **old.rank.diagnostic.exact_thresholds(policy, d, cut))
                    else:
                        full = {**{k: policy[k] for k in ("cv", "forced_add", "forced_joint")},
                                "ols4": p["ols4"], **pair.release(policy, t)}
                        local[m].extend(pair.assess(full, d, cut))
                if role == "held":
                    change.extend(contrast(p, d))
                    counts.update(item["held"]["counts"])
            beat(phase="score", fold=name)
        for m in METHODS:
            sites[m].append(pair.summarize(local[m]))
            rows[m].extend({**r, "record": f"{site}:{r['record']}"} for r in local[m])
        diffs[site] = dict(pair=report.effects.site_means([(r["record"], r["pair"]) for r in change]),
            trajectory=report.effects.site_means([(r["record"], r["trajectory_lower_pp"]) for r in change
                                                  if r["trajectory_lower_pp"] is not None]))
    totals = {m: old.rank.metrics(rows[m], sites[m]) for m in METHODS}
    expected = totals["feasibility8"]["forced_joint"]["edits"]
    if expected != totals["joint8"]["forced_joint"]["edits"]:
        raise ValueError("Forced slot coverage changed")
    out = dict(registration=key, freeze=study.artifact(path), result_source="fresh_run", new_training=False,
        held_counts=dict(counts), held_score=totals, score_breakpoints=thresholds, by_fold=captures,
        capture_totals={r: dict(sum((Counter(v[r]) for v in captures.values()), Counter())) for r in ("cal", "held")},
        direct_pair_native=report.effects.interval([r["pair"] for r in diffs.values()]),
        direct_trajectory_lower_pp=report.effects.interval([r["trajectory"] for r in diffs.values()]),
        direct_by_location=diffs, independent_confirmation=False, reserved_outcomes_opened=False, original_2pct=True)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("capture_totals", "direct_pair_native", "held_counts")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent = register()
        if phase == "freeze":
            freeze(key, cohorts, parent)
        else:
            score(key, cohorts)
        beat(phase=phase+"_complete")


if __name__ == "__main__":
    main()
