"""Cross-fit a learned FIFO continuation value on exposed OpenDD sites."""

import argparse
from collections import Counter
import fcntl
from itertools import combinations
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
from scripts import run_m3w_opendd_release_credit as prior
from src.world_model import m3w_opendd_credit_value as model

run, parent, two, paired = prior.run, prior.parent, prior.two, prior.paired
candidate, hard = prior.prior, prior.prior.prior
OUT = prior.OUT.parent/"opendd_credit_value_v2"
PRIVATE = candidate.PRIVATE.parent/OUT.name
STEPS = 400
ARMS = ("immediate", "myopic", "continuation")


def register():
    k, a, old_heads, old, source, _, _, _ = prior.register()
    files = [prior.OUT/"analysis.json", candidate.OUT/"producer_freeze.json", hard.OUT/"head_freeze.json"]
    past, producers, hardness = map(run.read, files)
    if (past["registration"] != k or past["confirmation_access"] or not past["original_2pct"]
            or producers["registration"] != run.sha(candidate.OUT/"registration.json")
            or not producers["outer_sites_excluded"] or old_heads != hardness["heads"]):
        raise ValueError("Require verified release and double-excluded forecast lineage")
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in files],
        native_steps=[8, 12], seeds=two.study.prior.SEEDS, threads=4, workers=0, steps=STEPS,
        features="current ranked causal row summaries and lagged query summary; FIFO state; no outcomes or final length",
        target="two action contrasts: immediate utility, and optimal offline continuation under the same FIFO clock",
        reward="lower CV gain minus positive all/easy/signed excess with weights 2/4/1; normalized by observed motion",
        expiry=model.EXPIRY, rank="fixed causal CV/OLS disagreement for all new policies",
        fitting="donor actions use candidate and release heads excluding both donor and outer site; no held fitting; donor cells with no discretionary actions are logged and omitted",
        control="same two-output network, no continuation at inference for myopic; match released credit IDs for direct utility",
        no_threshold_search=True, original_2pct=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(reg), a, old_heads, old, source, past, producers["predictions"]


@torch.inference_mode()
def packets(d, cv, ols, ego, starts, release_head):
    x, scale = candidate.model.features(d["h"], cv, ego)
    drift = hard.prior.model.instability(d["h"], cv, ols)
    legal = (cv != ego).any((1, 2))
    probability = release_head(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()
    count = hard.model.decide(probability, legal)
    v = np.c_[x, probability, np.log1p(np.maximum(drift, 0.))]
    q, rec = d["query"], d["rec"]
    order = np.argsort(q, kind="stable")
    bound = np.searchsorted(q[order], np.arange(q.max()+2))
    for r in np.unique(rec):
        query = np.unique(q[rec == r])
        if not (np.diff(starts[query]) > 0).all():
            raise ValueError("Broken query clock")
        last = np.zeros(v.shape[1]+3)
        rows, features, scales = [], [], []
        for t in query:
            at = order[bound[t]:bound[t+1]]
            local = at[legal[at]]
            n = int(count[at].sum())
            if not (rec[at] == r).all():
                raise ValueError("Query spans recordings")
            rank = local[np.argsort(-drift[local], kind="stable")]
            best = rank[:n+1]
            size = float(np.median(scale[at]))
            now = np.r_[v[at].mean(0), np.log1p([len(at), len(local), size])]
            z = v[best]
            summary = np.r_[np.zeros((1, 2*v.shape[1])),
                            np.c_[z.cumsum(0)/np.arange(1, len(z)+1)[:, None], z]]
            rows.append(best)
            features.append((np.r_[now, last], summary, n))
            scales.append(size)
            last = now
        yield rows, features, scales


def row_rewards(d, cv, ego, easy_cut):
    y, mask = d["y"], d["valid"]
    m = two.moments(cv, ego, y, mask, easy_cut)
    cost = np.column_stack([m[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    return paired(cv, ego, y, mask)[1].mean(1) - cost.clip(min=0.) @ [2., 4., 1.]


def training_rows(d, cv, ols, ego, starts, head, cut):
    x, y = [], []
    reward = row_rewards(d, cv, ego, cut)
    for rows, features, scales in packets(d, cv, ols, ego, starts, head):
        labels, _ = model.action_targets([p[2] for p in features], [np.r_[0., reward[r].cumsum()] for r in rows], scales)
        for p, states in zip(features, labels, strict=True):
            for state, target in states.items():
                if len(target) > 1:
                    x.append(model.feature_rows(p, state)[1:])
                    y.append(target[1:].clip(-12., 12.))
    if not x:
        return None
    return torch.from_numpy(np.concatenate(x)), torch.from_numpy(np.concatenate(y).astype(np.float32))


@torch.inference_mode()
def evaluate(d, bank, ego, starts, release_head, head, cut, previous):
    cv, ols = bank[:2]
    y, mask = d["y"], d["valid"]
    pred = {k: cv.copy() for k in (*ARMS, "myopic_matched", "continuation_matched")}
    log = {k: Counter() for k in ARMS}
    for rows, features, _ in packets(d, cv, ols, ego, starts, release_head):
        counts = np.array([p[2] for p in features])
        immediate = np.concatenate([r[:n] for r, n in zip(rows, counts, strict=True)])
        pred["immediate"][immediate] = ego[immediate]
        log["immediate"]["edits"] += len(immediate)
        for arm, j in (("myopic", 0), ("continuation", 1)):
            score = lambda x: head(torch.from_numpy(x)).numpy()[:, j]
            events, lost = model.stream(iter(zip(rows, features, strict=True)), score)
            e = np.asarray(events, dtype=int).reshape(-1, 4)
            emitted = np.bincount(e[:, 2], minlength=len(rows))
            if (len(events)+lost != len(immediate) or len(np.unique(e[:, 1])) != len(events)
                    or len(np.unique(e[:, 0])) != len(events) or not ((e[:, 3] >= 0) & (e[:, 3] <= 2)).all()
                    or not prior.model.legal_schedule(counts, emitted, [len(r) for r in rows], 2, lost == 0)):
                raise ValueError("Learned policy exceeded the same causal edit budget")
            pred[arm][e[:, 0]], pred[arm+"_matched"][e[:, 1]] = ego[e[:, 0]], ego[e[:, 1]]
            log[arm].update(edits=len(e), delayed=int((e[:, 3] > 0).sum()), expired=lost)
    stats = {k: parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    if stats["immediate"]["edits"] != previous["metrics"]["immediate"]["edits"]:
        raise ValueError("Original held-site release count did not reproduce")
    for k in ARMS[1:]:
        if stats[k]["edits"] != stats[k+"_matched"]["edits"] or stats[k]["edits"] != log[k]["edits"]:
            raise ValueError("Credit-matched immediate control has a different count")
    mass = np.linalg.norm(cv-y, axis=-1)[mask].sum()
    direct = {(a+"_vs_"+b): hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, mask, mass)
              for a in ARMS[1:] for b in ("cv", "immediate", a+"_matched")}
    direct["continuation_vs_myopic"] = hard.paired_pp(pred["myopic"], pred["continuation"], y, mask, mass)
    return dict(metrics=stats, schedule=log, effects=direct)


def execute(key, a, old_heads, old, source, past, receipts, replay, beat):
    sites = {s: candidate.forecast.load(s, source)[:3] for s in parent.source.SITES}
    bank = {s: parent.prior.analytic(v[0]) for s, v in sites.items()}
    seeds = two.study.prior.SEEDS
    cut = lambda use: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in use]), [.25, .75])
    x = {s: torch.from_numpy(hard.model.features(v[0]["h"])) for s, v in sites.items()}
    pred = {(tuple(v["excluded"]), v["target"], v["seed"]): v for v in receipts}
    width = x[next(iter(x))].shape[1]
    updates, freeze = Counter(), {}
    for p in combinations(sorted(sites), 2):
        donors = [s for s in sites if s not in p]
        fit = []
        for s in donors:
            label, known = hard.labels(sites[s][0], bank[s][0], cut([v for v in donors if v != s]))
            fit.append((x[s][known], torch.from_numpy(label[known])))
        for seed in seeds:
            name = "release_"+"_".join(p)+f"_{seed}"
            ident = dict(registration=key, excluded=list(p), fit=donors, seed=seed)
            _, freeze[name], u = two.fit.fit(ident, name, seed, lambda: hard.model.network(width),
                                              hard.STEPS, two.batcher(fit), hard.model.loss, replay, beat)
            updates["release"] += u
    results = {}
    for held, (d, _, starts) in sites.items():
        donors = [s for s in sites if s != held]
        results[held] = []
        for seed in seeds:
            train, donor_rows = [], {}
            for s in donors:
                pair = tuple(sorted((held, s)))
                h = freeze["release_"+"_".join(pair)+f"_{seed}"]
                if {s, held} & set(h["identity"]["fit"]):
                    raise ValueError("Release head exposed to donor or evaluation site")
                rec = pred[pair, s, seed]
                ego = np.load(run.checked(rec["prediction"]), allow_pickle=False)
                rel = two.fit.load(h, lambda: hard.model.network(width), hard.STEPS)
                batch = training_rows(sites[s][0], *bank[s][:2], ego, sites[s][2], rel,
                                      cut([v for v in donors if v != s])[0])
                donor_rows[s] = 0 if batch is None else len(batch[0])
                if batch is not None:
                    train.append(batch)
            if not train:
                raise ValueError("No fitting action at any donor site")
            name = f"value_{held}_{seed}"
            ident = dict(registration=key, held=held, fit=donors, seed=seed)
            n = train[0][0].shape[1]
            head, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n), STEPS,
                                                 two.batcher(train), model.loss, replay, beat)
            updates["value"] += u
            rel = two.fit.load(old_heads[f"{held}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            row = next(v for v in old["by_site"][held] if v["seed"] == seed)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                ego = z["comoving_ego"]
            prev = next(v for v in past["by_site"][held] if v["seed"] == seed)
            res = evaluate(d, bank[held], ego, starts, rel, head, cut(donors), prev)
            results[held].append(dict(seed=seed, donor_actions=donor_rows, **res))
            beat(phase="held", site=held, seed=seed, edits=res["schedule"])
    flat = [r for v in results.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        v = np.array([np.mean([r["effects"][k] for r in group]) for group in results.values()])
        return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(v.min()))

    summary = dict(effects={k: ci(k) for k in flat[0]["effects"]},
        schedule={k: sum((r["schedule"][k] for r in flat), Counter()) for k in ARMS},
        metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in prior.FIELDS} for k in flat[0]["metrics"]})
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=results, summary=summary,
        original_2pct=True, confirmation_access=False, submission_ready=False)))
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
        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if v.get("phase") != "fit" or v["step"] == STEPS:
                print(z, flush=True)
        two.fit.PRIVATE = PRIVATE
        updates = execute(*register(), a.replay, beat)
        beat(complete=True, updates=updates, replay=a.replay)


if __name__ == "__main__":
    main()
