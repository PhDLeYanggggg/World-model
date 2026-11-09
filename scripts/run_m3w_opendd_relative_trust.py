"""Test a causal relative-displacement cap at the original matched edit budget."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_joint_retrieval_debit as debit

parent, np, study, base = debit.run, debit.np, debit.study, debit.base
old, pair, torch = parent.old, parent.pair, parent.torch
OUT = parent.OUT.parent/"opendd_relative_trust_v1"
LOCAL = parent.LOCAL.parent/"opendd_relative_trust_v1"
FRACTION = .5
ARMS = ("joint8", "trust8")


def beat(**info):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), **info))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    prev_key, cohorts, fit = parent.register()
    p = parent.OUT/"summary.json"
    prev = study.read(p)
    frozen = study.read(study.checked(prev["freeze"]))
    if (prev["registration"] != prev_key or frozen["registration"] != prev_key
            or frozen["held_targets_used"] or prev["reserved_outcomes_opened"]):
        raise ValueError("Require verified parent retrieval forecasts")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(parent=study.artifact(p), parent_freeze=prev["freeze"],
        parent_debit=study.artifact(parent.OUT/"debit_audit.json"),
        code=[study.artifact(Path(x)) for x in (__file__, parent.__file__, debit.__file__, old.relative.__file__)],
        arms=ARMS, fraction=FRACTION, shortlist=parent.WIDTH, seeds=old.SEEDS, sites=old.SITES,
        treatment="candidate mean displacement of relative paths from common additive reference <= half uncapped selected slate's",
        ranking="same fusion and final ranking inside eligible candidates, never pad when fewer than eight",
        budget="identical two positive edit peaks vs CV per menu query; additive reference itself is legal",
        primary="known, debit and lower vs common reference on same full grid; also direct trust vs parent",
        calibration="same disjoint roles, original nonempty grid and exact breakpoint checks",
        new_training=False, independent_confirmation=False, original_2pct=True, reserved_outcomes_opened=False))
    return base.sha(reg), cohorts, fit, frozen


def pick(scores, shortlist):
    return int(shortlist[int(scores.argmax()) if scores.max() > 0 else 0])


def capped_shortlist(feasibility, gain, bound, original):
    if (not np.isfinite(bound).all() or (bound < 0).any()
            or len(bound) != len(feasibility) or not (bound == 0).any()):
        raise ValueError("Require a finite geometry bound and the legal additive reference")
    eligible = np.flatnonzero(bound <= FRACTION*bound[original])
    _, subset = parent.retrieve(feasibility[eligible], gain[eligible])
    return eligible[subset], eligible


@torch.inference_mode()
def infer(gain, feasibility, data, beta, control):
    for k, name in (("q", "q"), ("cv", "cv"), ("ols", "ols4")):
        np.testing.assert_array_equal(data[k], control[name])
    cv, q = data["cv"], data["q"]
    p = dict(q=q, cv=cv, ols4=data["ols"], forced_add=control["forced_add"], trust8=cv.copy(),
             score_trust8=np.full(len(cv), -1.), short_trust8=np.full((q.max()+1, parent.WIDTH), -1, dtype=int),
             bound_joint8=np.zeros(q.max()+1), bound_trust8=np.zeros(q.max()+1))
    counts = Counter()
    for ix, m in pair.menus(data):
        a, query = old.reference(m, beta), q[ix.start]
        np.testing.assert_array_equal(a, control["forced_add"][ix])
        f, u, b = [], [], []
        for k in range(0, len(m["opts"]), 256):
            x, g = pair.encode(m, m["opts"][k:k+256])
            f.append(feasibility(torch.from_numpy(x), torch.from_numpy(g)).sigmoid().numpy()[:, 1, 0])
        for k in range(0, len(m["opts"]), 128):
            options = m["opts"][k:k+128]
            v = pair.paths_for(m, options)
            x, xr, bound, _, _ = old.relative.features(m["h"], m["c"], m["ols"], v, a)
            z = gain(*(torch.from_numpy(t) for t in (x, xr, bound))).numpy()
            u.append(z[:, 0]-z[:, 1])
            b.append(bound)
        f, u, b = map(np.concatenate, (f, u, b))
        _, joint = parent.retrieve(f, u)
        old_short = control["short_joint8"][query]
        np.testing.assert_array_equal(joint, old_short[old_short >= 0])
        scores = parent.scored_gain(gain, m, pair.paths_for(m, m["opts"][joint]), a)
        original = pick(scores, joint)
        np.testing.assert_array_equal(pair.paths_for(m, m["opts"][original:original+1])[0], control["joint8"][ix])
        np.testing.assert_array_equal(f[original], control["score_joint8"][ix])
        shortlist, eligible = capped_shortlist(f, u, b, original)
        scores = parent.scored_gain(gain, m, pair.paths_for(m, m["opts"][shortlist]), a)
        choice = pick(scores, shortlist)
        prediction = pair.paths_for(m, m["opts"][choice:choice+1])[0]
        p["trust8"][ix] = prediction
        p["score_trust8"][ix] = f[choice]
        p["short_trust8"][query, :len(shortlist)] = shortlist
        p["bound_joint8"][query], p["bound_trust8"][query] = b[original], b[choice]
        peak = pair.run.turn.turn.peak(prediction-cv[ix])
        np.testing.assert_allclose(np.sort(peak[peak > 0]), np.sort(m["radius"]), atol=1e-8, rtol=1e-8)
        counts.update(queries=1, legal=len(b), eligible=len(eligible),
            short_under_eight=int(len(shortlist) < parent.WIDTH),
            reference_path=int(np.array_equal(prediction, a)),
            changed_from_parent=int(not np.array_equal(prediction, control["joint8"][ix])))
    return p, dict(counts)


def freeze(key, cohorts, fits, prev):
    inputs = {s: pair.causal(cohorts[s]) for s in old.SITES}
    result = {}
    for site in old.SITES:
        cal, train = pair.roles(site)
        for seed in old.SEEDS:
            name = f"{site}_{seed}"
            path = LOCAL/f"{name}.json"
            source, fit = prev["folds"][name], fits["folds"][name]
            if (source["cal_site"], source["held_site"], source["seed"]) != (cal, site, seed):
                raise ValueError("Control roles do not match")
            if not path.exists():
                g, f = parent.load_models(fit, site, seed, train, cal)
                roles = {}
                for role, s in (("cal", cal), ("held", site)):
                    control = old.rank.checked_npz(source[role]["forecast"])
                    p, counts = infer(g, f, inputs[s], np.asarray(fit["reference"]), control)
                    forecast = pair.run.turn.exp.previous.save_bank(LOCAL/f"{name}_{role}.npz", p)
                    roles[role] = dict(forecast=forecast, counts=counts, control=source[role]["forecast"])
                base.immutable(path, dict(registration=key, cal_site=cal, held_site=site,
                    seed=seed, cut=source["cut"], **roles))
            z = study.read(path)
            if (z["registration"], z["cal_site"], z["held_site"], z["seed"], z["cut"]) != (key, cal, site, seed, source["cut"]):
                raise ValueError("Forecast checkpoint identity changed")
            for role in ("cal", "held"):
                study.checked(z[role]["forecast"])
                if z[role]["control"] != source[role]["forecast"]:
                    raise ValueError("Parent forecast changed")
            result[name] = z
            beat(phase="freeze", done=len(result), fold=name, total=len(old.SITES)*len(old.SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=result, held_targets_used=False))


def opportunity(inp, d, p, cut):
    out = Counter()
    for ix, m in pair.menus(inp):
        q, a = p["q"][ix.start], p["forced_add"][ix]
        y, v = d["y"][ix], d["v"][ix]
        qualify, allowed = [], []
        for k in range(0, len(m["opts"]), 128):
            z = pair.paths_for(m, m["opts"][k:k+128])
            _, _, bound, _, _ = old.relative.features(m["h"], m["c"], m["ols"], z, a)
            qualify.extend(old.target.targets(m, z, a, y, v, cut)[1])
            allowed.extend(bound <= FRACTION*p["bound_joint8"][q])
        qualify, allowed = np.asarray(qualify), np.asarray(allowed)
        short = p["short_trust8"][q]
        short = short[short >= 0]
        if not allowed[short].all() or len(short) != min(parent.WIDTH, allowed.sum()):
            raise ValueError("Illegal or incomplete constrained shortlist")
        out.update(unconstrained=int(qualify.any()), under_cap=int((qualify & allowed).any()),
            retrieved=int(qualify[short].any()), selected=int(old.target.targets(m, p["trust8"][ix][None], a, y, v, cut)[1][0]))
    return dict(out)


def load_role(z, role):
    p = old.rank.checked_npz(z[role]["forecast"])
    c = old.rank.checked_npz(z[role]["control"])
    for k in ("q", "cv", "ols4", "forced_add"):
        np.testing.assert_array_equal(p[k], c[k])
    p.update({k: c[k] for k in ("joint8", "score_joint8")})
    return p


def comparisons(p, d):
    fields = {"parent_vs_add": ("joint8", "forced_add"), "trust_vs_add": ("trust8", "forced_add"),
              "trust_vs_parent": ("trust8", "joint8")}
    for rec in np.unique(d["rec"]):
        at = d["rec"] == rec
        q, y, v = (d[k][at] for k in ("q", "y", "v"))
        if not (p["joint8"][at] != p["cv"][at]).any() or not debit.frontier.denominators(q, v).all():
            continue
        terms = {k: debit.components(p[a][at], p[b][at], y, v, q) for k, (a, b) in fields.items()}
        # The difference of bounds relative to a common anchor is not
        # the conservative lower bound for the direct head-to-head.
        terms["common_anchor_change"] = (np.asarray(terms["trust_vs_add"])-terms["parent_vs_add"]).tolist()
        direct = parent.report.direct(dict(cv=p["cv"][at], feasibility=p["joint8"][at], pair_gain=p["trust8"][at]),
                                      dict(q=q, y=y, v=v))
        yield str(rec), {**terms, "direct_trajectory_lower_pp": direct["trajectory_lower_pp"]}


def score(key, cohorts):
    path = OUT/"freeze.json"
    frozen = study.read(path)
    folds = frozen["folds"]
    if (frozen["registration"] != key or frozen["held_targets_used"]
            or set(folds) != {f"{s}_{j}" for s in old.SITES for j in old.SEEDS}):
        raise ValueError("Need all frozen folds")
    data = pair.run.turn.exp.loader(cohorts)
    inp = {s: pair.causal(cohorts[s]) for s in old.SITES}
    thresholds, capacity = {}, {}
    for name, z in folds.items():
        cal = z["cal_site"]
        p = load_role(z, "cal")
        capacity[name] = {"cal": opportunity(inp[cal], data[cal], p, z["cut"][0])}
        thresholds[name] = {}
        for a in ARMS:
            policy = old.as_policy(p, a)
            thresholds[name][a] = dict(grid=pair.choose_threshold(policy, data[cal], z["cut"])[0],
                **old.rank.diagnostic.exact_thresholds(policy, data[cal], z["cut"]))
        beat(phase="calibrate", fold=name)
    receipt = OUT/"calibration.json"
    base.immutable(receipt, dict(registration=key, freeze=study.artifact(path), folds=thresholds))
    rows, sites = ({a: [] for a in ARMS} for _ in range(2))
    effects, counts = {}, Counter()
    for site in old.SITES:
        local, delta = {a: [] for a in ARMS}, []
        for seed in old.SEEDS:
            name = f"{site}_{seed}"
            z, d = folds[name], data[site]
            p = load_role(z, "held")
            for k in ("q", "cv"):
                np.testing.assert_array_equal(p[k], d[k])
            for a in ARMS:
                policy = old.as_policy(p, a)
                full = {**{k: policy[k] for k in ("cv", "forced_add", "forced_joint")},
                        "ols4": p["ols4"], **pair.release(policy, thresholds[name][a]["grid"])}
                local[a].extend(pair.assess(full, d, z["cut"]))
            counts.update(z["held"]["counts"])
            capacity[name]["held"] = opportunity(inp[site], d, p, z["cut"][0])
            delta.extend(comparisons(p, d))
            beat(phase="score", fold=name)
        for a in ARMS:
            sites[a].append(pair.summarize(local[a]))
            rows[a].extend({**r, "record": f"{site}:{r['record']}"} for r in local[a])
        effects[site] = {k: debit.effects.site_means([(r, v[k]) for r, v in delta if v[k] is not None])
                         for k in delta[0][1]} if delta else {}
    metrics = {a: old.rank.metrics(rows[a], sites[a]) for a in ARMS}
    if metrics["joint8"]["forced_joint"]["edits"] != metrics["trust8"]["forced_joint"]["edits"]:
        raise ValueError("Intervention count changed")
    out = dict(registration=key, freeze=study.artifact(path), calibration=study.artifact(receipt),
        result_source="fresh_run", checkpoints="cached_verified", new_training=False,
        metrics=metrics, counts=dict(counts), capacity_by_fold=capacity,
        capacity={role: dict(sum((Counter(r[role]) for r in capacity.values()), Counter())) for role in ("cal", "held")},
        contrasts={k: debit.effects.interval([z.get(k) for z in effects.values()])
                   for k in sorted({k for z in effects.values() for k in z})},
        component_order=["known", "missing_debit", "lower"], effects_by_location=effects,
        calibration_totals={a: {k: sum(z[a][k] for z in thresholds.values())
            for k in ("nonempty_breakpoints", "feasible_breakpoints")} for a in ARMS},
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "capacity", "contrasts", "calibration_totals")}, indent=2))


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
        key, cohorts, fits, prev = register()
        if phase == "freeze":
            freeze(key, cohorts, fits, prev)
        else:
            score(key, cohorts)
        beat(phase=phase+"_complete")


if __name__ == "__main__":
    main()
