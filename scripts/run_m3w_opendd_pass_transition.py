"""Learn the change in recording pass probability on the same support menu."""

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
from scripts import run_m3w_opendd_support_directions as prior_study
from src.world_model import m3w_opendd_pass_transition as model

run, record, v, two, hard, prior, early, wait, old = (prior_study.run, prior_study.record, prior_study.v,
    prior_study.two, prior_study.hard, prior_study.prior, prior_study.early, prior_study.wait, prior_study.old)
OUT = prior_study.OUT.parent/"opendd_pass_transition_v1"
PRIVATE = prior_study.PRIVATE.parent/OUT.name
ARMS = ("original", "composite", "transition", "rescue_only", "composite_matched", "same_row_composite")
READOUTS = ("transition", "rescue_only")
STEPS = 600
QUANTILE = .90


def register():
    key, *inputs = prior_study.register()
    a = early.timing.legacy.read(prior_study.OUT.name)
    early.timing.read_support_directions_attribution()
    h = run.read(prior_study.OUT/"head_freeze.json")
    if a["registration"] != key or h["registration"] != key:
        raise ValueError("Require frozen expanded support study")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(prior_study.OUT/f"{k}.json")
            for k in ("registration", "analysis", "head_freeze", "attribution_registration", "attribution_analysis")],
        steps=STEPS, seeds=two.study.prior.SEEDS, threads=4, workers=0, native_steps=[8, 12],
        target="explicit unconditional rescue, destruction and reference loss events; outcomes only in labels",
        sampling="uniform donor site, recording, candidate",
        readout="P(rescue)-P(destroy); rescue-only ablation from same head; reference loss logit negative in both",
        gate="90th percentile of train-record maxima, floored at zero; not independent risk calibration",
        controls="causal frozen composite; retrospective record-max composite at matched additions/replacements; same row composite direction",
        menu="same expanded support directions and exact radius; same original gain and support priority",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), *inputs, a, h["heads"]


def training(menu, groups):
    x = menu[2]
    data, count = [], Counter()
    for _, _, _, _, rows, k, t, ok, original in groups:
        if not len(rows):
            continue
        y = model.targets(original, ok, t[:, 1] > .5)
        data.append((x[k, rows], y))
        count.update(records=1, actions=len(rows), rescue=int(y[:, 0].sum()), destruction=int(y[:, 1].sum()))
    return data, count


def batcher(donors):
    def batch(rng):
        sizes = rng.multinomial(128, np.full(len(donors), 1./len(donors)))
        x, y = [], []
        for data, size in zip(donors, sizes, strict=True):
            for i in rng.integers(len(data), size=size):
                a, b = data[i]
                j = rng.integers(len(a))
                x.append(a[j])
                y.append(b[j])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))
    return batch


@torch.inference_mode()
def cutoffs(net, donors):
    maxima = {k: [] for k in READOUTS}
    for data in donors:
        lengths = [len(x) for x, _ in data]
        p = model.probabilities(net(torch.from_numpy(np.concatenate([x for x, _ in data]))).numpy())
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        for k in READOUTS:
            maxima[k].extend(np.maximum.reduceat(model.advantage(p, k == "rescue_only"), starts))
    return {k: max(0., float(np.quantile(a, QUANTILE))) for k, a in maxima.items()}


@torch.inference_mode()
def evaluate(d, cv, ols, cuts, pack, menu, groups, net, frozen, gates, before):
    large, small = pack[:2]
    bank, _, x, _ = menu
    flat = torch.from_numpy(x.reshape(-1, x.shape[-1]))
    p = model.probabilities(net(flat).numpy()).reshape(*x.shape[:2], 3)
    z = frozen(flat).numpy().reshape(*x.shape[:2], 2)
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    choices, matched_pool, oracle_labels = {}, {True: [], False: []}, {}
    count = Counter()
    for r, g, b, _, rows, k, _, feasible, original in groups:
        score = z[k, rows]
        event = p[k, rows]
        take = prior_study.model.first_release
        choices[r] = {
            "composite": take(rows, k, q[rows], score[:, 0], score[:, 1] < 0., before["cutoffs"]["expanded"]),
            **{a: take(rows, k, q[rows], model.advantage(event, a == "rescue_only"),
                       event[:, 2] < .5, gates[a]) for a in READOUTS}}
        current = choices[r]["transition"]
        if len(rows):
            # Retrospective budget control, with no labels in its ranking.
            i = int(np.lexsort((-score[:, 0], score[:, 1] >= 0.))[0])
            matched_pool[b is None].append((bool(score[i, 1] >= 0.), -float(score[i, 0]), r, (int(rows[i]), int(k[i]))))
        if current is not None:
            count["transition_new" if b is None else "transition_advanced"] += 1
            same = np.flatnonzero(rows == current[0])
            j = same[np.lexsort((-score[same, 0], score[same, 1] >= 0.))[0]]
            choices[r]["same_row_composite"] = int(rows[j]), int(k[j])
        else:
            choices[r]["same_row_composite"] = None
        choices[r]["composite_matched"] = None
        oracle_labels[r] = original, {(int(i), int(j)): bool(ok) for i, j, ok in zip(rows, k, feasible, strict=True)}
        count["opportunities"] += int(not original and feasible.any())
        if not original and feasible.any() and current is None:
            count["transition_missed_no_release"] += 1
    for add, key in ((True, "transition_new"), (False, "transition_advanced")):
        for _, _, r, candidate in sorted(matched_pool[add])[:count[key]]:
            choices[r]["composite_matched"] = candidate

    pred = {k: cv.copy() for k in ARMS}
    for r, g, b, *_ in groups:
        if g is not None:
            for p0 in pred.values():
                p0[g] = large[g]
        if b is not None:
            pred["original"][b] = small[b]
        original, feasible = oracle_labels[r]
        for a, choice in choices[r].items():
            if choice is None:
                if b is not None:
                    pred[a][b] = small[b]
                continue
            i, j = choice
            if i == g or (g is not None and q[i] > q[g]) or (b is not None and q[i] >= q[b]):
                raise ValueError("Proposal violates the frozen support and gain releases")
            pred[a][i] = bank[j, i]
            joint = feasible[choice]
            count.update({a+"_rescued": int(not original and joint), a+"_destroyed": int(original and not joint),
                          a+"_interventions": 1})
    stats = {k: v.parent.score(d, cv, ols, a, cuts) for k, a in pred.items()}
    for k, b in (("original", "original"), ("composite", "expanded")):
        if any(stats[k][f] != before["metrics"][b][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen composite/original did not replay")
    for k in ARMS[1:]:
        if stats[k]["dual_positive"] != stats["original"]["dual_positive"]+count[k+"_rescued"]-count[k+"_destroyed"]:
            raise ValueError("Event labels do not reconcile with original score")
    for k in ("composite_matched", "same_row_composite"):
        if (stats[k]["edits"] != stats["transition"]["edits"]
                or count[k+"_interventions"] != count["transition_interventions"]):
            raise ValueError("Control did not match intervention counts")
    risk = {k: early.policy(a, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]
            for k, a in pred.items()}
    for k in ARMS[1:]:
        for r, before0 in risk["original"].items():
            count[k+"_lost_reference"] += int(before0["completion_easy_risk_upper"] is not None
                                               and risk[k][r]["completion_easy_risk_upper"] is None)
    denom = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, "original") for k in ARMS[1:]]
    pairs += [("transition", k) for k in ("composite", "composite_matched", "same_row_composite", "rescue_only")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, denom) for a, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=count)


def execute(key, outer, old_forecasts, source, gains, _rank, producers, releases, nested, _protected, _wait,
            frozen, frozen_heads, parent, parent_heads, replay, beat):
    sites = {s: wait.study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
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
            h = two.fit.load(frozen_heads[name], lambda: early.model.network(p[-2].shape[1]), old.STEPS)
            cutoff = lookup(frozen)["donor_threshold"][0]
            fit, donor = [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                r = releases["release_"+"_".join(pair)+f"_{seed}"]
                f = nested[f"first_{held}_{s}_{seed}"]
                if (set(r["identity"]["fit"]) & set(pair) or set(f["identity"]["excluded"]) != set(pair)
                        or set(f["identity"]["fit"]) & set(pair)):
                    raise ValueError("Upstream nominations expose held or donor labels")
                ds, (cv, ols) = sites[s], banks[s][:2]
                z = prepare(s, record.read_prediction(refs, pair, s, seed), r, f)
                m = prior_study.menu(ds, cv, ols, z)
                roster = prior_study.ref.proposals(z, h, cutoff)
                examples = prior_study.cases(ds, cv, ols, cuts([t for t in use if t != s]), z, m, roster)
                train, info = training(m, examples)
                if train:
                    fit.append(train)
                donor.update(info)
            n = fit[0][0][0].shape[1]
            ident = dict(registration=key, held=held, fit=use, seed=seed)
            net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n), STEPS,
                batcher(fit), model.loss, replay, beat)
            updates += u
            gates = cutoffs(net, fit)
            cv, ols = banks[held][:2]
            m = prior_study.menu(d, cv, ols, p)
            cut = cuts(use)
            roster = prior_study.ref.proposals(p, h, cutoff)
            examples = prior_study.cases(d, cv, ols, cut, p, m, roster)
            frozen_net = two.fit.load(parent_heads[name+"_expanded"], lambda: prior_study.model.network(n),
                                      prior_study.STEPS)
            z = evaluate(d, cv, ols, cut, p, m, examples, net, frozen_net, gates, lookup(parent))
            result[held].append(dict(seed=seed, donor=donor, cutoffs=gates, **z))
            beat(phase="site", site=held, seed=seed, updates=updates, rescues=donor["rescue"], destruction=donor["destruction"])
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
