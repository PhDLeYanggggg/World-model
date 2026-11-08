"""Learn and evaluate recipient slates at fixed and adaptive edit times."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_credit_value as v
from src.world_model import m3w_opendd_slate_queue as m

run, two, hard, c = v.run, v.two, v.hard, v.candidate
OUT, PRIVATE = v.OUT.parent/"opendd_slate_queue_v1", v.PRIVATE.parent/"opendd_slate_queue_v1"
ARMS = ("fixed_now", "adaptive_now", "fixed_queue", "myopic_queue", "adaptive_queue", "no_geometry")
MATCHED = ("same_clock", "same_credits_immediate")
STEPS = 500


def register():
    k, _, outer, previous, source, timing, producers = v.register()
    files = [v.OUT/f"{n}.json" for n in ("analysis", "head_freeze", "prefix_capacity", "prefix_registration")]
    a, h, p, pr = map(run.read, files)
    if (a["registration"] != k or a["heads"] != run.sha(files[1]) or h["registration"] != k
            or p["registration"] != run.sha(files[3]) or not pr["exact_original_2pct"]
            or any(z["confirmation_access"] for z in (a, p, pr))):
        raise ValueError("Frozen timing dependencies changed")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(m.__file__)),
        parents=[run.artifact(p) for p in files], seeds=two.study.prior.SEEDS, native_steps=[8, 12],
        threads=4, workers=0, steps=STEPS, models=["slate", "geometry_ablated"], arms=ARMS,
        candidates=["CV/OLS disagreement", "large candidate change", "small candidate change",
                    "observed turn", "nearest observed neighbor"],
        targets="myopic and hindsight continuation contrasts over legal slates; fixed old expiry and risk proxy",
        geometry="relative rollout separation on selected-selected and selected-unselected edges, in past scales",
        matching="same_clock matches exact query counts; same_credits_immediate matches FIFO IDs and per-recording counts",
        primary="adaptive_queue vs fixed recipient prefix at identical realized query counts",
        training="all fitting candidate and release heads exclude outer site and donor site",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        original_2pct=True, no_threshold_search=True, confirmation_access=False)))
    return run.sha(f), outer, previous, source, timing, producers, h["heads"], p["by_site"]


@torch.inference_mode()
def packets(d, cv, ols, ego, starts, head):
    x, scale = c.model.features(d["h"], cv, ego)
    drift = hard.prior.model.instability(d["h"], cv, ols)
    p = head(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()
    legal = (cv != ego).any((1, 2))
    released = hard.model.decide(p, legal)
    h, query, rec = (d[k] for k in ("h", "query", "rec"))
    deviation = np.linalg.norm(cv-ego, axis=-1).max(1)/scale
    early, late = h[:, 3]-h[:, 0], h[:, -1]-h[:, -4]
    turn = 1.-(early*late).sum(1)/np.maximum(np.linalg.norm(early, axis=1)*np.linalg.norm(late, axis=1), 1e-8)
    x = np.c_[x, p, np.log1p(drift), np.log1p(deviation), turn]
    order = np.argsort(query, kind="stable")
    end = np.searchsorted(query[order], np.arange(query.max()+2))
    for r in np.unique(rec):
        q = np.unique(query[rec == r])
        if not (np.diff(starts[q]) > 0).all():
            raise ValueError("Expected chronological queries")
        previous = np.zeros(x.shape[1]+3)
        sequence = []
        for t in q:
            at = order[end[t]:end[t+1]]
            n = int(released[at].sum())
            size = float(np.median(scale[at]))
            origin = h[at, -1]
            distance = np.linalg.norm(origin[:, None]-origin[None, :], axis=-1)
            np.fill_diagonal(distance, np.inf)
            local = np.flatnonzero(legal[at])
            subset, fixed = m.menu(local, (drift[at], deviation[at], -deviation[at], turn[at], -distance.min(1)),
                                   min(n+1, len(local)))
            g = m.geometry(h[at], cv[at], ego[at], subset, size)
            now = np.r_[x[at].mean(0), np.log1p([len(at), len(local), size])]
            selected = np.asarray([x[at[z]].mean(0) if len(z) else np.zeros(x.shape[1]) for z in subset])
            other = np.asarray([(x[at].sum(0)-selected[i]*len(z))/max(len(at)-len(z), 1)
                                for i, z in enumerate(subset)])
            f = np.c_[np.broadcast_to(np.r_[now, previous], (len(subset), 2*len(now))), selected, other, g]
            sequence.append(m.Packet(tuple(at[z] for z in subset), f.astype(np.float32), fixed,
                                     np.asarray([len(z) for z in subset]), n, size))
            previous = now
        yield r, sequence


def training(d, cv, ols, ego, starts, head, cut):
    y = v.row_rewards(d, cv, ego, cut)
    rows = [a for _, p in packets(d, cv, ols, ego, starts, head) for a in m.training(p, y)]
    if not rows:
        return None
    x, z = (np.concatenate(a) for a in zip(*rows, strict=True))
    return torch.from_numpy(x), torch.from_numpy(z.astype(np.float32))


def exact(seq, gain, risk, easy, ols0, mass, delay):
    count = np.array([p.release for p in seq])
    if not count.sum():
        return "no_credits", None
    sets = [z for p in seq for z in p.rows[1:]]
    qid = np.repeat(np.arange(len(seq)), [len(p.rows)-1 for p in seq])
    n = np.asarray([len(z) for z in sets])
    horizon = np.arange(len(seq))[:, None]
    same = horizon == qid
    total = count.cumsum()
    due = np.r_[np.zeros(delay), total][:len(seq)].copy()
    active = n <= (total-np.r_[0., due[:-1]])[qid]
    due[-1] = total[-1]
    support = np.asarray([easy[z].sum() for z in sets])
    if not (active & (support > 0)).any():
        return "no_reachable_easy", None
    g = np.asarray([gain[z].sum(0) for z in sets])
    r = np.asarray([risk[z].sum(0) for z in sets])
    zero = np.zeros(len(seq))
    M = np.r_[np.c_[same, zero], np.c_[(horizon >= qid)*n, zero],
              np.c_[r.T, np.zeros(3)], np.c_[-g.T, np.ones(2)], np.r_[support, 0.][None]]
    low = np.r_[zero, due, [-np.inf]*5, 1.]
    high = np.r_[np.ones(len(seq)), total, 0., 0., .02*mass, 0., ols0, np.inf]
    scale = np.maximum(np.abs(M).max(1), 1.)
    s = milp(np.r_[np.zeros(len(sets)), -1.], integrality=np.r_[np.ones(len(sets)), 0.],
        bounds=Bounds(np.r_[np.zeros(len(sets)), -np.inf], np.r_[active.astype(float), np.inf]),
        constraints=LinearConstraint(M/scale[:, None], low/scale, high/scale),
        options=dict(time_limit=5., mip_rel_gap=0.))
    if s.status == 2:
        return "risk_infeasible", None
    if s.x is not None:
        on = s.x[:-1] > .5
        used = np.bincount(qid[on], weights=n[on], minlength=len(seq))
        if ((same[:, on].sum(1) <= 1).all() and support[on].sum() > 0 and
                (r[on].sum(0) <= [0., 0., .02*mass]).all() and (g[on].sum(0)+[0., ols0]).min() > 0. and
                v.prior.model.legal_schedule(count, used, [len(p.fixed)-1 for p in seq], delay, True)):
            return "qualified", np.concatenate([sets[i] for i in np.flatnonzero(on)])
    bound = getattr(s, "mip_dual_bound", None)
    return ("not_dual_positive" if bound is not None and np.isfinite(bound) and -bound < -1e-8 else "unresolved"), None


@torch.inference_mode()
def evaluate(d, bank, ego, starts, release, heads, cut, ref, prefix, full):
    cv, ols = bank[:2]
    q, y, mask = (d[k] for k in ("query", "y", "valid"))
    pred = {k: cv.copy() for k in (*ARMS, *MATCHED)}
    clock, caps = {k: Counter() for k in ARMS}, {str(i): Counter() for i in (0, 2)}
    mom = two.moments(cv, ego, y, mask, cut[0])
    risk = np.column_stack([mom[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    ols0 = v.paired(ols, cv, y, mask)[1].sum(1)
    gain = np.c_[v.paired(cv, ego, y, mask)[1].sum(1), v.paired(ols, ego, y, mask)[1].sum(1)-ols0]
    same_count_changes = 0
    for r, seq in packets(d, cv, ols, ego, starts, release):
        count = np.array([p.release for p in seq])
        for arm in ARMS:
            net = heads[arm == "no_geometry"]
            score = lambda x: net(torch.from_numpy(x)).numpy()
            plan, lost = m.stream(iter(seq), score, arm)
            spend = np.array([len(z[0]) for z in plan])
            used = [token for _, _, tokens in plan for token in tokens]
            if (len(used) != len(set(used)) or spend.sum()+lost != count.sum() or
                    not v.prior.model.legal_schedule(count, spend, [len(p.fixed)-1 for p in seq], 2, lost == 0)):
                raise ValueError("Slate plan violated release budget")
            clock[arm].update(edits=int(spend.sum()), expired=lost,
                              delayed=sum(t != j for t, (_, _, z) in enumerate(plan) for j, _ in z))
            for at, _, _ in plan:
                pred[arm][at] = ego[at]
            if arm == "adaptive_queue":
                matched = np.bincount([j for j, _ in used], minlength=len(seq))
                for p, k, (at, fixed, _) in zip(seq, matched, plan, strict=True):
                    pred["same_clock"][fixed] = ego[fixed]
                    same_count_changes += len(set(at)-set(fixed))
                    ix, x = m.actions(p, (0, 0))
                    same = np.flatnonzero(p.size[ix] == k)
                    pick = p.rows[ix[same[score(x[same])[:, 0].argmax()]]]
                    pred["same_credits_immediate"][pick] = ego[pick]
        at = np.flatnonzero(d["rec"] == r)
        for delay in (0, 2):
            status, chosen = exact(seq, gain, risk, mom["easy"], ols0[at].sum(), mom["ref_easy"][at].sum(), delay)
            if status == "qualified":
                local = cv[at].copy()
                local[np.searchsorted(at, chosen)] = ego[chosen]
                stat = v.parent.score(dict(y=y[at], valid=mask[at], rec=d["rec"][at]), cv[at], ols[at], local, cut)
                if stat["dual_positive"] != 1 or stat["edits"] != count.sum():
                    raise ValueError("Exact slate witness does not satisfy original scorer")
            caps[str(delay)][status] += 1
    edits = {k: np.bincount(q[(p != cv).any((1, 2))], minlength=q.max()+1) for k, p in pred.items()}
    np.testing.assert_array_equal(edits["adaptive_queue"], edits["same_clock"])
    np.testing.assert_array_equal(edits["adaptive_now"], edits["fixed_now"])
    for r in np.unique(d["rec"]):
        at = np.unique(q[d["rec"] == r])
        if edits["adaptive_queue"][at].sum() != edits["same_credits_immediate"][at].sum():
            raise ValueError("Matched immediate must have the same recording total")
    for t, cap in caps.items():
        if not (prefix[t].get("qualified", 0) <= cap["qualified"]+cap["unresolved"] and
                cap["qualified"] <= full[t].get("qualified", 0)+full[t].get("unresolved", 0)):
            raise ValueError("Menu capacity should lie between fixed and unrestricted recipients")
    metrics = {k: v.parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    for f in v.prior.FIELDS:
        if metrics["fixed_now"][f] != ref["metrics"]["immediate"][f]:
            raise ValueError("Fixed immediate baseline changed")
    mass = np.linalg.norm(cv-y, axis=-1)[mask].sum()
    pairs = [("adaptive_queue", k) for k in ("cv", *MATCHED, "fixed_queue", "no_geometry", "myopic_queue")]
    pairs.append(("adaptive_now", "fixed_now"))
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, mask, mass) for a, b in pairs}
    rel = c.relative_effect(d, pred["same_clock"], pred["adaptive_queue"])
    return dict(metrics=metrics, schedule=clock, capacity=caps, effects=effects, pair_relative=rel,
                same_clock_recipients_changed=same_count_changes)


def execute(key, outer, previous, source, timing, producers, release, prefix, replay, beat):
    sites = {s: c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    bank = {s: v.parent.prior.analytic(a[0]) for s, a in sites.items()}
    pred = {(tuple(z["excluded"]), z["target"], z["seed"]): z for z in producers}
    width = hard.model.features(next(iter(sites.values()))[0]["h"][:1]).shape[1]
    cuts = lambda use: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in use]), [.25, .75])
    full = run.read(v.prior.OUT/"analysis.json")["by_site"]
    parent_result = run.read(v.OUT/"analysis.json")["by_site"]
    result, freeze, updates = {}, {}, 0
    for held, (d, _, starts) in sites.items():
        fit = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            train = []
            for s in fit:
                pair = tuple(sorted((held, s)))
                h = release["release_"+"_".join(pair)+f"_{seed}"]
                if set(h["identity"]["fit"]) & set(pair):
                    raise ValueError("Donor release head has site overlap")
                producer = pred[pair, s, seed]
                path = np.load(run.checked(producer["prediction"]), allow_pickle=False)
                rel = two.fit.load(h, lambda: hard.model.network(width), hard.STEPS)
                b = training(sites[s][0], *bank[s][:2], path, sites[s][2], rel, cuts([w for w in fit if w != s])[0])
                if b is not None:
                    train.append(b)
            heads = []
            for geo in (True, False):
                name = f"{held}_{seed}_{'full' if geo else 'no_geometry'}"
                ident = dict(registration=key, held=held, fit=fit, seed=seed, geometry=geo)
                data = train if geo else [(m.remove_geometry(x.clone()), y) for x, y in train]
                net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: v.model.network(train[0][0].shape[1]),
                                                   STEPS, two.batcher(data), v.model.loss, replay, beat)
                heads.append(net)
                updates += u
            row = next(z for z in previous["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            r = two.fit.load(outer[f"{held}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            pick = lambda rows: next(z for z in rows[held] if z["seed"] == seed)
            a = evaluate(d, bank[held], path, starts, r, heads, cuts(fit), pick(parent_result),
                         pick(prefix)["counts"], pick(full)["capacity"])
            result[held].append(dict(seed=seed, **a))
            beat(phase="site", site=held, seed=seed, updates=updates)
    flat = [r for rows in result.values() for r in rows]
    rng = np.random.default_rng(817)

    def ci(fn):
        x = [[fn(r) for r in rows] for rows in result.values()]
        vals = np.array([np.mean(z) for z in x if all(t is not None for t in z)])
        if not len(vals):
            return None
        draw = rng.integers(len(vals), size=(3000, len(vals)))
        return dict(point=float(vals.mean()), interval95=np.quantile(vals[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(vals.min()), sites=len(vals))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in flat[0]["metrics"]},
        schedule={k: sum((r["schedule"][k] for r in flat), Counter()) for k in ARMS},
        capacity={t: sum((r["capacity"][t] for r in flat), Counter()) for t in ("0", "2")},
        effects={k: ci(lambda r: r["effects"][k]) for k in flat[0]["effects"]},
        pair_relative=ci(lambda r: r["pair_relative"]),
        same_clock_recipients_changed=sum(r["same_clock_recipients_changed"] for r in flat))
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    a = cli.parse_args()
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
