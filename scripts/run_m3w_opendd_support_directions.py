"""Expand early support directions without increasing the candidate radii."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import run_m3w_opendd_reference_loss as ref
from src.world_model import m3w_opendd_support_directions as model

run, record, v, two, hard, prior, early, wait, old = ref.run, ref.record, ref.v, ref.two, ref.hard, ref.prior, ref.early, ref.wait, ref.old
OUT = ref.OUT.parent/"opendd_support_directions_v1"
PRIVATE = ref.PRIVATE.parent/OUT.name
HEADS = ("neural", "expanded")
ORACLES = ("trigger_neural", "trigger_expanded", "window_neural", "window_expanded")
ARMS = ("original", "frozen", *HEADS, "same_row_neural", *ORACLES)
STEPS = 500
QUANTILE = .90


def register():
    key, *inputs = ref.register()
    a = early.timing.legacy.read(ref.OUT.name)
    early.timing.read_reference_loss_oracle()
    if a["registration"] != key:
        raise ValueError("Require frozen reference-loss parent")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(ref.OUT/f"{k}.json")
            for k in ("registration", "analysis", "head_freeze", "oracle_registration", "oracle_analysis")],
        steps=STEPS, seeds=two.study.prior.SEEDS, threads=4, workers=0, native_steps=[8, 12],
        bank="native small neural path; OLS4 and past-prefix selected direction scaled down to the same peak radius",
        clock="gain is frozen; no supplemental support after original gain or at/after original support",
        risk_target="transition in joint pass, signed conservative utility and direct loss of unique old easy reference",
        gate="first positive score above donor-record maximum quantile .90; loss logit must be negative; no held outcomes",
        controls="retrained neural-only; exact expanded query/row/radius/count replay with neural direction",
        audit="exact any admissible choice at frozen proposal row or early window; retain original if jointly passing",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), *inputs


def menu(d, cv, ols, pack):
    small, radius, scale = pack[1], pack[5], pack[-1]
    past = wait.shape.analytic(d["h"])[2]
    bank, legal, matched = model.match(cv, small, (ols, past))
    np.testing.assert_allclose(matched, radius, rtol=1e-10, atol=1e-12)
    common = ref.describe(pack)
    features = []
    for k, p in enumerate(bank):
        geometry = wait.study.c.model.features(d["h"], cv, p)[0]
        features.append(np.c_[common, geometry, np.broadcast_to(np.eye(3)[k], (len(common), 3))])
        np.testing.assert_allclose(wait.shape.peak((p-cv)[legal[k]]), radius[legal[k]], rtol=1e-10, atol=1e-12)
    return bank, legal, np.stack(features).clip(-16., 16.).astype(np.float32), scale


def cases(d, cv, ols, cuts, pack, bank_menu, roster):
    large, small = pack[:2]
    bank, legal, _, scale = bank_menu
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    sm = record.contributions(cv, ols, small, y, valid, cuts[0])
    big = record.contributions(cv, ols, large, y, valid, cuts[0])
    cs = [record.contributions(cv, ols, p, y, valid, cuts[0]) for p in bank]
    terms = [np.stack(z) for z in zip(*(c[:4] for c in cs), strict=True)]
    add = np.stack([v.paired(cv, p, y, valid)[1].sum(1) for p in bank])
    remove = v.paired(small, cv, y, valid)[1].sum(1)
    out = []
    for r, g, b, j in roster:
        at = np.flatnonzero(rec == r)
        # This is equivalent to observing the two original triggers in sequence.
        use = at[at != g]
        if g is not None:
            use = use[q[use] <= q[g]]
        if b is not None:
            use = use[q[use] < q[b]]
        k, ix = np.nonzero(legal[:, use])
        rows = use[ix]
        fixed = tuple(x[g] for x in big[:4]) if g is not None else None
        mass, offset = sm[2][at].sum(), sm[4][at].sum()
        before = [t[b:b+1] if b is not None else np.zeros_like(t[:1]) for t in sm[:4]]
        prior_risk, prior_joint = early.support_targets(*before, mass, fixed, offset)
        parts = [t[k, rows] for t in terms]
        risk, joint = early.support_targets(*parts, mass, fixed, offset)
        lost = model_ref_loss(sm, big, parts[2], g, b)
        utility = (add[k, rows]+(remove[b] if b is not None else 0.))/(12.*scale[rows])
        target = (joint.astype(float)-prior_joint.astype(float)-.5*(prior_risk & ~risk)
                  -.5*lost+.1*utility.clip(-1., 1.))
        out.append((r, g, b, j, rows, k, np.c_[target, lost].astype(np.float32),
                    joint & ~lost, bool(prior_joint[0])))
    return out


def model_ref_loss(sm, big, candidate_ref, g, b):
    old_unique = b is not None and sm[2][b] > 0. and (g is None or big[2][g] <= 0.)
    return old_unique & (candidate_ref <= 0.)


def data(bank_menu, groups, arm):
    xs, ys, sizes = [], [], []
    x = bank_menu[2]
    for _, _, _, _, rows, k, target, _, _ in groups:
        use = k == 0 if arm == "neural" else np.ones(len(k), bool)
        if use.any():
            xs.append(x[k[use], rows[use]])
            ys.append(target[use])
            sizes.append(int(use.sum()))
    if not sizes:
        return None
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))), sizes


@torch.inference_mode()
def evaluate(d, cv, ols, cuts, pack, bank_menu, groups, nets, gates, previous):
    large, small = pack[:2]
    bank, _, x, _ = bank_menu
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    pred = {k: cv.copy() for k in ARMS}
    z = {k: net(torch.from_numpy(x.reshape(-1, x.shape[-1]))).numpy().reshape(*x.shape[:2], 2)
         for k, net in nets.items()}
    count = Counter()
    for r, g, b, j, rows, k, target, feasible, prior_joint in groups:
        if g is not None:
            for p in pred.values():
                p[g] = large[g]
        edits = {a: (b, 0) for a in ARMS}
        edits["frozen"] = j, 0
        for arm in HEADS:
            score = z[arm][k, rows]
            legal = (score[:, 1] < 0.) & ((k == 0) if arm == "neural" else True)
            choice = model.first_release(rows, k, q[rows], score[:, 0], legal, gates[arm])
            if choice is not None:
                edits[arm] = choice
                count.update({arm+"_new": int(b is None), arm+"_advanced": int(b is not None),
                              arm+"_alternative_direction": int(choice[1] > 0)})
        edits["same_row_neural"] = edits["expanded"][0], 0
        if not prior_joint:
            trigger = (rows == j) if j is not None and j != b else np.zeros(len(rows), bool)
            for arm, allow in zip(ORACLES, (trigger & (k == 0), trigger, k == 0, np.ones(len(k), bool)), strict=True):
                hit = np.flatnonzero(feasible & allow)
                if len(hit):
                    i = hit[0]
                    edits[arm] = int(rows[i]), int(k[i])
                    count[arm+"_new_joint"] += 1
        for a, (s, f) in edits.items():
            if s is not None:
                if s == g or (s != b and (g is not None and q[s] > q[g])):
                    raise ValueError("A support candidate changed the original gain")
                pred[a][s] = bank[f, s]
    stats = {a: v.parent.score(d, cv, ols, p, cuts) for a, p in pred.items()}
    for a, b in (("original", "original"), ("frozen", "frozen")):
        if any(stats[a][f] != previous["metrics"][b][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen control did not reproduce")
    if stats["expanded"]["edits"] != stats["same_row_neural"]["edits"]:
        raise ValueError("Matched shape comparison changed the budget")
    for a in ORACLES:
        if stats[a]["dual_positive"] != stats["original"]["dual_positive"]+count[a+"_new_joint"]:
            raise ValueError("Exact menu witness failed original scoring")
    if not (stats["trigger_neural"]["dual_positive"] <= stats["trigger_expanded"]["dual_positive"]
            <= stats["window_expanded"]["dual_positive"]
            and stats["trigger_neural"]["dual_positive"] <= stats["window_neural"]["dual_positive"]
            <= stats["window_expanded"]["dual_positive"]):
        raise ValueError("The expanded action menu must contain the old menu")
    risk = {a: early.policy(p, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]
            for a, p in pred.items()}
    for a in ARMS[1:]:
        for r in risk["original"]:
            old_ref = risk["original"][r]["completion_easy_risk_upper"] is not None
            new_ref = risk[a][r]["completion_easy_risk_upper"] is not None
            count[a+"_lost_reference"] += int(old_ref and not new_ref)
    if any(count[a+"_lost_reference"] for a in ORACLES):
        raise ValueError("Oracle must preserve previously defined references")
    denom = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(a, "cv") for a in ARMS]+[(a, "original") for a in ARMS[1:]]
    pairs += [("expanded", "neural"), ("expanded", "same_row_neural")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, denom) for a, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=count)


def execute(key, outer, old_forecasts, source, gains, _rank, producers, releases, nested, _protected, _wait,
            frozen, heads, replay, beat):
    sites = {s: wait.study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    parent = early.timing.legacy.read(ref.OUT.name)
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0

    def prepare(s, path, r, f):
        d, cv = sites[s], banks[s][0]
        release = two.fit.load(r, lambda: hard.model.network(width), hard.STEPS)
        n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
        return early.prepare(d, cv, path, release, two.fit.load(f, lambda: prior.model.network(n), prior.STEPS))

    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            name = f"{held}_{seed}"
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(lookup(old_forecasts)["prediction"]), allow_pickle=False) as z:
                p = prepare(held, z["comoving_ego"], outer[name], gains[name])
            h = two.fit.load(heads[name], lambda: early.model.network(p[-2].shape[1]), old.STEPS)
            cutoff = lookup(frozen)["donor_threshold"][0]
            fit, donor = {a: [] for a in HEADS}, Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                r = releases["release_"+"_".join(pair)+f"_{seed}"]
                f = nested[f"first_{held}_{s}_{seed}"]
                if (set(r["identity"]["fit"]) & set(pair) or set(f["identity"]["excluded"]) != set(pair)
                        or set(f["identity"]["fit"]) & set(pair)):
                    raise ValueError("Upstream nominations expose held or donor labels")
                ds, (cv, ols) = sites[s], banks[s][:2]
                z = prepare(s, record.read_prediction(refs, pair, s, seed), r, f)
                m = menu(ds, cv, ols, z)
                r = ref.proposals(z, h, cutoff)
                examples = cases(ds, cv, ols, cuts([t for t in use if t != s]), z, m, r)
                for arm in HEADS:
                    data0 = data(m, examples, arm)
                    if data0 is not None:
                        fit[arm].append(data0)
                        donor[arm] += sum(data0[1])
            gates, nets = {}, {}
            for arm in HEADS:
                tag = name+"_"+arm
                ident = dict(registration=key, held=held, fit=use, seed=seed, arm=arm)
                n = fit[arm][0][0][0].shape[1]
                net, freeze[tag], u = two.fit.fit(ident, tag, seed, lambda: model.network(n), STEPS,
                    two.batcher([a for a, _ in fit[arm]]), model.loss, replay, beat)
                nets[arm] = net
                updates += u
                with torch.inference_mode():
                    maxima = np.concatenate([np.maximum.reduceat(net(x).numpy()[:, 0], np.r_[0, np.cumsum(sizes)[:-1]])
                                             for (x, _), sizes in fit[arm]])
                gates[arm] = max(0., float(np.quantile(maxima, QUANTILE)))
            cv, ols = banks[held][:2]
            m = menu(d, cv, ols, p)
            cut = cuts(use)
            r = ref.proposals(p, h, cutoff)
            examples = cases(d, cv, ols, cut, p, m, r)
            z = evaluate(d, cv, ols, cut, p, m, examples, nets, gates, lookup(parent))
            result[held].append(dict(seed=seed, donor=donor, cutoffs=gates, **z))
            beat(phase="site", site=held, seed=seed, updates=updates, candidates=donor["expanded"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if p.get("phase") != "fit":
                print(z, flush=True)

        two.fit.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
