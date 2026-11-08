"""Test a separate counterfactual reference-loss guard on frozen early proposals."""

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
from scripts import run_m3w_opendd_early_advantage as old
from src.world_model import m3w_opendd_reference_loss as model

run, record, v, two, hard, prior, early, wait = old.run, old.record, old.v, old.two, old.hard, old.prior, old.early, old.wait
OUT = old.OUT.parent/"opendd_reference_loss_v1"
PRIVATE = old.PRIVATE.parent/OUT.name
GUARDS = ("direct", "factorized")
ARMS = ("original", "frozen", *GUARDS, *(k+"_easy_matched" for k in GUARDS))
STEPS = 500
KEEP = .75


def register():
    key, *inputs = old.register()
    a = early.timing.legacy.read(old.OUT.name)
    parents = [old.OUT/f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    h = run.read(parents[-1])
    if a["registration"] != key or h["registration"] != key:
        raise ValueError("Require frozen early-advantage parent")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in parents],
        steps=STEPS, threads=4, workers=0, native_steps=[8, 12], seeds=two.study.prior.SEEDS,
        base="frozen early advantage proposal, exact gain and early proposal paths",
        target="remove original unique easy reference and add no new easy reference",
        arms="direct loss probability and factored probability with conditional no-reference head",
        gate="keep if below donor in-sample proposal quantile; empty donor proposals reject all",
        keep_quantile=KEEP, refusal="disable further supplemental proposals; original support can still fire",
        control="retroactive same count per held location, seed and addition/replacement class by original easy score",
        primary="direct vs frozen and direct easy-matched; factorization diagnostic",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), *inputs, a, h["heads"]


def describe(pack):
    _, _, blocks, gain, easy, _, x, scale = pack
    context = np.zeros((len(x), 5), dtype=np.float32)
    for seq in blocks.values():
        high_easy, high_gain, observed = 0., 0., 0
        for t, rows in enumerate(seq):
            context[rows] = np.c_[np.broadcast_to(
                [np.log1p(t), high_easy, high_gain, np.log1p(observed), np.log1p(len(rows))],
                (len(rows), 5))]
            if len(rows):
                high_easy = max(high_easy, float(easy[rows].max()))
                high_gain = max(high_gain, float((gain[rows]/scale[rows]).max()))
                observed += len(rows)
    return np.c_[x, context].clip(-16., 16.).astype(np.float32)


@torch.inference_mode()
def proposals(pack, net, cutoff):
    _, _, blocks, gain, easy, radius, x, _ = pack
    z = net(torch.from_numpy(x)).numpy()[:, 0]-cutoff
    threshold = wait.clock.base.model.EASY_THRESHOLD
    out = []
    for r, seq in blocks.items():
        g, b, _ = wait.clock.base.model.choose((ix, gain[ix], easy[ix]) for ix in seq)
        a, j, _ = early.model.choose(((ix, gain[ix], easy[ix], radius[ix], z[ix], np.ones(len(ix)))
                                       for ix in seq), threshold, "mass")
        if a != g or (j is None and b is not None):
            raise ValueError("Frozen proposal no longer preserves gain or fallback")
        out.append((r, g, b, j))
    return out


def training(d, cv, cut, pack, x, roster):
    small, radius = pack[1], pack[5]
    rec, q, y, valid = (d[k] for k in ("rec", "query", "y", "valid"))
    easy = two.moments(cv, small, y, valid, cut)["ref_easy"] > 0.
    rows, labels, proposed, counts = [], [], [], Counter()
    for r, g, b, j in roster:
        use = np.flatnonzero((rec == r) & (radius > 0.) & (np.arange(len(rec)) != g))
        if g is not None:
            use = use[q[use] <= q[g]]
        if b is not None:
            use = use[q[use] < q[b]]
        if j != b and j is not None:
            if j not in use:
                raise ValueError("Frozen early candidate outside causal menu")
            proposed.append(j)
        if not len(use):
            continue
        t = model.reference_targets(g is not None and easy[g], b is not None and easy[b], easy[use])
        rows.append(use)
        labels.append(t)
        counts.update(rows=len(use), lost=int(t[:, 0].sum()), old_unique=int(t[:, 1].sum()))
    if not rows:
        return None, np.asarray(proposed, int), counts
    data = (torch.from_numpy(x[np.concatenate(rows)]), torch.from_numpy(np.concatenate(labels).astype(np.float32)))
    return data, np.asarray(proposed, int), counts


@torch.inference_mode()
def evaluate(d, cv, ols, cuts, pack, x, roster, net, cutoffs, ref):
    large, small, _, _, easy, _, _, _ = pack
    q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
    changed = [(r, g, b, j) for r, g, b, j in roster if j != b and j is not None]
    ix = np.asarray([j for _, _, _, j in changed], int)
    p = model.scores(net(torch.from_numpy(x[ix])).numpy())
    masks = dict(original=np.zeros(len(ix), bool), frozen=np.ones(len(ix), bool))
    adds = np.asarray([b is None for _, _, b, _ in changed], dtype=bool)
    for i, k in enumerate(GUARDS):
        masks[k] = p[:, i] <= cutoffs[i]
        shadow = np.zeros(len(ix), dtype=bool)
        for group in (adds, ~adds):
            shadow[group] = model.matched_easy(easy[ix[group]], masks[k][group])
        masks[k+"_easy_matched"] = shadow
    predicted = {}
    for k, keep in masks.items():
        switched = {r: j for (r, _, _, j), use in zip(changed, keep, strict=True) if use}
        output = cv.copy()
        for r, g, b, _ in roster:
            j = switched.get(r, b)
            if g is not None:
                output[g] = large[g]
            if j is not None:
                if j == g or (j != b and g is not None and q[j] > q[g]):
                    raise ValueError("Guard violates fixed gain order")
                output[j] = small[j]
        predicted[k] = output
    stats = {k: v.parent.score(d, cv, ols, z, cuts) for k, z in predicted.items()}
    for k, b in (("original", "original"), ("frozen", "mass")):
        if any(stats[k][f] != ref["metrics"][b][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen early policy or original score did not replay")
    risk = {k: early.policy(z, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]
            for k, z in predicted.items()}
    count = Counter()
    count["proposals"] = len(ix)
    for (r, _, b, _), i in zip(changed, range(len(ix)), strict=True):
        old_ref = risk["original"][r]["completion_easy_risk_upper"] is not None
        candidate_ref = risk["frozen"][r]["completion_easy_risk_upper"] is not None
        lost = old_ref and not candidate_ref
        restored = candidate_ref and not old_ref
        for k, keep in masks.items():
            if keep[i]:
                count.update({k+"_accepted": 1, k+"_added": int(b is None), k+"_replaced": int(b is not None),
                              k+"_lost_reference": int(lost), k+"_new_reference": int(restored)})
    for k in GUARDS:
        if (masks[k].sum() != masks[k+"_easy_matched"].sum()
                or stats[k]["edits"] != stats[k+"_easy_matched"]["edits"]):
            raise ValueError("Failed to match admitted proposals")
    if any(stats[k]["edits"] != stats["original"]["edits"]+count[k+"_added"] for k in ARMS):
        raise ValueError("A guard changed a frozen action count")
    if count["frozen_lost_reference"] != ref["diagnostics"].get("mass_lost_reference", 0):
        raise ValueError("Frozen reference loss did not replay")
    pairs = [(k, "cv") for k in ARMS]+[(k, "original") for k in ARMS[1:]]
    pairs += [(k, "frozen") for k in GUARDS]+[(k, k+"_easy_matched") for k in GUARDS]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else predicted[b], predicted[a], y, valid, mass)
               for a, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=count)


def execute(key, outer, old_forecasts, source, gains, _rank, producers, releases, nested, _protected, _wait,
            frozen, heads, replay, beat):
    sites = {s: wait.study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0

    def pack(s, path, r, f):
        d, cv = sites[s], banks[s][0]
        release = two.fit.load(r, lambda: hard.model.network(hard_width), hard.STEPS)
        n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
        return early.prepare(d, cv, path, release, two.fit.load(f, lambda: prior.model.network(n), prior.STEPS))

    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            name = f"{held}_{seed}"
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(lookup(old_forecasts)["prediction"]), allow_pickle=False) as f:
                held_pack = pack(held, f["comoving_ego"], outer[name], gains[name])
            n = held_pack[-2].shape[1]
            frozen_head = two.fit.load(heads[name], lambda: early.model.network(n), old.STEPS)
            cutoff = lookup(frozen)["donor_threshold"][0]
            fit, proposal_x, donor = [], [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                r = releases["release_"+"_".join(pair)+f"_{seed}"]
                f = nested[f"first_{held}_{s}_{seed}"]
                if (set(r["identity"]["fit"]) & set(pair) or set(f["identity"]["excluded"]) != set(pair)
                        or set(f["identity"]["fit"]) & set(pair)):
                    raise ValueError("Upstream nominations expose held or donor labels")
                p = pack(s, record.read_prediction(refs, pair, s, seed), r, f)
                x = describe(p)
                a, j, info = training(sites[s], banks[s][0], cuts([t for t in use if t != s])[0], p, x,
                                      proposals(p, frozen_head, cutoff))
                if a is not None:
                    fit.append(a)
                if len(j):
                    proposal_x.append(x[j])
                donor.update(info)
            n = fit[0][0].shape[1]
            ident = dict(registration=key, held=held, fit=use, seed=seed)
            net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n), STEPS,
                                              two.batcher(fit), model.loss, replay, beat)
            updates += u
            with torch.inference_mode():
                z = model.scores(net(torch.from_numpy(np.concatenate(proposal_x))).numpy()) if proposal_x else None
            cutoffs = np.quantile(z, KEEP, axis=0) if z is not None else np.zeros(2)
            p = held_pack
            z = evaluate(d, *banks[held][:2], cuts(use), p, describe(p), proposals(p, frozen_head, cutoff),
                         net, cutoffs, lookup(frozen))
            result[held].append(dict(seed=seed, donor=donor, donor_proposals=sum(map(len, proposal_x)),
                                     cutoffs=cutoffs.tolist(), **z))
            beat(phase="site", site=held, seed=seed, updates=updates, rows=donor["rows"], losses=donor["lost"])
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
