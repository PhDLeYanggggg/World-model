"""Learn from feasible recording assignments without changing the held clock."""

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
from scripts import run_m3w_opendd_slate_rank as prior
from src.world_model import m3w_opendd_record_teacher as model

study, v, run, two, hard, m = prior.study, prior.v, prior.run, prior.two, prior.hard, prior.m
OUT, PRIVATE = prior.OUT.parent/"opendd_record_teacher_v1", prior.PRIVATE.parent/"opendd_record_teacher_v1"
ARMS = ("fixed", "scalar", "guarded", "local_teacher", "record_teacher")
STEPS = 500


def register():
    k, outer, old, source, producers, releases, clock, _ = prior.register()
    files = [prior.OUT/f"{n}.json" for n in ("analysis", "head_freeze")]
    a, h = map(run.read, files)
    if (a["registration"] != k or h["registration"] != k or a["heads"] != run.sha(files[1])
            or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require prior same-clock controls")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(p) for p in files], steps=STEPS, seeds=two.study.prior.SEEDS,
        threads=4, workers=0, native_steps=[8, 12], arms=ARMS,
        training="fixed immediate and one-query-delay donor clocks, using release and trajectory heads excluding donor and outer site",
        teacher="only positive-dual-gain original-risk witnesses; no label for an unsupported donor clock",
        matched_teacher="same donor clocks, query options and examples; independent maximum lower CV gain in each query",
        primary="record teacher vs local teacher at every frozen held query count; original risk and gain separately",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        original_2pct=True, no_threshold_search=True, confirmation_access=False)))
    return run.sha(f), outer, old, source, producers, releases, clock, h["heads"], a


def clocks(seq):
    count = np.array([p.release for p in seq])
    yield count
    rank = [p.rows[p.fixed[-1]] for p in seq]
    packets = [(rows, p.release, -np.arange(len(rows)), np.zeros(len(rows))) for rows, p in zip(rank, seq, strict=True)]
    spent, lost = v.prior.model.stream(iter(packets), "fixed_wait")
    delayed = np.bincount([t for _, _, t, _ in spent], minlength=len(seq))
    if (delayed.sum()+lost != count.sum() or
            not v.prior.model.legal_schedule(count, delayed, [len(p.fixed)-1 for p in seq], 2, lost == 0)):
        raise ValueError("Donor delay control violates the original clock")
    if not np.array_equal(count, delayed):
        yield delayed


def training(d, cv, ols, path, starts, head, cut):
    y, mask, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    mom = two.moments(cv, path, y, mask, cut[0])
    risk = np.column_stack([mom[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    base = v.paired(ols, cv, y, mask)[1].sum(1)
    low = v.paired(cv, path, y, mask)[1].sum(1)
    gain = np.c_[low, v.paired(ols, path, y, mask)[1].sum(1)-base]
    x, labels, status = [], [], Counter()
    for r, seq in study.packets(d, cv, ols, path, starts, head):
        at = np.flatnonzero(rec == r)
        for count in clocks(seq):
            fixed = [model.at_count(p, int(n)) for p, n in zip(seq, count, strict=True)]
            kind, rows = study.exact(fixed, gain, risk, mom["easy"], base[at].sum(), mom["ref_easy"][at].sum(), 0)
            status[kind] += 1
            if kind != "qualified":
                continue
            pred = cv[at].copy()
            pred[np.searchsorted(at, rows)] = path[rows]
            true = v.parent.score(dict(y=y[at], valid=mask[at], rec=rec[at]), cv[at], ols[at], pred, cut)
            queries = np.unique(q[at])
            if true["dual_positive"] != 1 or not np.array_equal(
                    np.bincount(np.searchsorted(queries, q[rows]), minlength=len(seq)), count):
                raise ValueError("Donor witness failed original risk or count checks")
            for p, n in zip(seq, count, strict=True):
                if n:
                    item = model.labels(p, n, rows, low)
                    if item is not None:
                        feats, target = item
                        x.append(feats)
                        labels.append(target)
                        status["example"] += 1
                        status["teacher_disagrees"] += int(target[0] != target[1])
    if not x:
        return None, status
    return (torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(labels, dtype=np.int64))), status


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, clock, old, heads, cuts, ref):
    cv, ols = bank[:2]
    y, mask, q = (d[k] for k in ("y", "valid", "query"))
    pred = {k: cv.copy() for k in ARMS}
    different = Counter()
    for _, seq in study.packets(d, cv, ols, path, starts, release):
        plan, _ = m.stream(iter(seq), lambda x: clock(torch.from_numpy(x)).numpy(), "adaptive_queue")
        for p, (used, fixed, _) in zip(seq, plan, strict=True):
            if not len(used):
                continue
            ids = np.flatnonzero(p.size == len(used))
            anchor = int(np.flatnonzero(ids == p.fixed[len(used)])[0])
            x = torch.from_numpy(p.features[ids])
            s, r = (net(x).numpy() for net in old)
            choices = dict(scalar=prior.model.decision(s, anchor), guarded=prior.model.decision(r, anchor, True))
            choices.update({k: int(net(x).numpy()[:, 0].argmax())
                            for k, net in zip(ARMS[-2:], heads, strict=True)})
            pred["fixed"][fixed] = path[fixed]
            for k, j in choices.items():
                ix = p.rows[ids[j]]
                pred[k][ix] = path[ix]
                different[k] += len(set(ix)-set(fixed))
            different["record_vs_local"] += len(set(p.rows[ids[choices["record_teacher"]]])-
                                                set(p.rows[ids[choices["local_teacher"]]]))
    counts = lambda p: np.bincount(q[(p != cv).any((1, 2))], minlength=q.max()+1)
    for p in pred.values():
        np.testing.assert_array_equal(counts(p), counts(pred["fixed"]))
    stats = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    for k in ARMS[:3]:
        for f in v.prior.FIELDS:
            if stats[k][f] != ref["metrics"][k][f]:
                raise ValueError("Frozen comparison changed")
    mass = np.linalg.norm(cv-y, axis=-1)[mask].sum()
    comparisons = [("record_teacher", b) for b in ("local_teacher", "fixed", "scalar", "guarded", "cv")]
    comparisons += [("local_teacher", b) for b in ("fixed", "cv")]
    gain = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, mask, mass) for a, b in comparisons}
    return dict(metrics=stats, effects=gain, changed=different)


def execute(key, outer, old, source, producers, releases, clock, prior_heads, ref, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    bank = {s: v.parent.prior.analytic(z[0]) for s, z in sites.items()}
    pairs = {(tuple(p["excluded"]), p["target"], p["seed"]): p for p in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))[0]["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0
    for held, (d, _, starts) in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit, donor = [], Counter()
            for s in use:
                pair = tuple(sorted((s, held)))
                h = releases["release_"+"_".join(pair)+f"_{seed}"]
                if set(h["identity"]["fit"]) & set(pair):
                    raise ValueError("Donor clock used held or donor labels")
                rel = two.fit.load(h, lambda: hard.model.network(hard_width), hard.STEPS)
                path = np.load(run.checked(pairs[pair, s, seed]["prediction"]), allow_pickle=False)
                b, c = training(sites[s][0], *bank[s][:2], path, sites[s][2], rel, cuts([u for u in use if u != s]))
                donor.update(c)
                if b is not None:
                    fit.append(b)
            if not fit:
                raise ValueError("No original-risk donor witnesses with alternatives")
            n = fit[0][0].shape[-1]
            heads = []
            for j, kind in enumerate(ARMS[-2:]):
                name = f"{held}_{seed}_{kind}"
                ident = dict(registration=key, held=held, fit=use, seed=seed, teacher=kind)
                net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n), STEPS,
                    two.batcher(fit), lambda net, x, y: model.loss(net, x, y, j), replay, beat)
                updates += u
                heads.append(net)
            rel = two.fit.load(outer[f"{held}_{seed}"], lambda: hard.model.network(hard_width), hard.STEPS)
            old_clock = two.fit.load(clock[f"{held}_{seed}_full"], lambda: v.model.network(n+m.QUEUE_WIDTH), study.STEPS)
            old_rank = [two.fit.load(prior_heads[f"{held}_{seed}_{kind}"], lambda: prior.model.network(n, kind == "scalar"),
                                     prior.STEPS) for kind in ("scalar", "paired")]
            row = next(z for z in old["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            before = next(z for z in ref["by_site"][held] if z["seed"] == seed)
            cell = evaluate(d, bank[held], path, starts, rel, old_clock, old_rank, heads, cuts(use), before)
            result[held].append(dict(seed=seed, donor=donor, **cell))
            beat(phase="site", site=held, seed=seed, updates=updates, teacher_disagreement=donor["teacher_disagrees"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        donors=sum((r["donor"] for r in flat), Counter()), changed=sum((r["changed"] for r in flat), Counter()))
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
            if p.get("phase") != "fit" or p["step"] == STEPS:
                print(z, flush=True)
        two.fit.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
