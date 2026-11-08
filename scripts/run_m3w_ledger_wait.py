"""Compare local versus actual-ledger deferral supervision on identical donors."""

import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_donor_risk_readout as donor_run
from scripts import run_m3w_release_wait as prior
from scripts.run_m3w_recording_swap_rank import load_pair
from src.world_model import m3w_ledger_wait as model

base, parent, run, study, torch = prior.base, prior.parent, prior.run, prior.study, prior.torch
timing = prior.prior
OUT, PRIVATE = prior.OUT.parent / "ledger_wait_v2", prior.PRIVATE.parent / "ledger_wait_v2"
STEPS = 800
MODES = ("local", "ledger")
ARMS = ("anchor", "old_risk", *(f"{m}_{g}" for m in MODES for g in ("gain", "risk")))


def register():
    key, a, ph, h, previous = prior.register()
    p, f = (prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze"))
    prev, frozen = run.read(p), run.read(f)
    dp = donor_run.OUT / "head_freeze.json"
    da = donor_run.OUT / "analysis.json"
    rp = study.OUT / "head_freeze.json"
    pair, pa, ranks = run.read(dp), run.read(da), run.read(rp)
    if (prev["registration"] != key or frozen["registration"] != key or prev["heads"] != run.sha(f)
            or prev["confirmation_access"] or not prev["original_2pct"]
            or pa["heads"] != run.sha(dp) or pa["registration"] != pair["registration"] or pa["confirmation_access"]):
        raise ValueError("Require the frozen release and pair-excluded evidence")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(z) for z in (p, f, da, dp, rp)], steps=STEPS, seeds=study.prior.SEEDS,
        primary="local and closed-loop difference labels at identical donor release states, features and optimizer",
        background="all keep, observed-prefix wait, and reproducible random 25% wait",
        donor_policy="pair-excluded nomination and factorized gain-only count under original token rules",
        difference="toggle one release, preserve all other deferral flags, compare whole recording; no hindsight actions at inference",
        previous_aborted=run.artifact(OUT.parent / "ledger_wait_v1" / "registration.json"),
        retry="normalize tuple/list checkpoint metadata with the existing verified pair loader",
        delay=model.AGE, native_steps=[8, 12], original_2pct=True, confirmation_access=False, no_held_tuning=True)))
    return run.sha(reg), a, ph, h, previous, prev, frozen["heads"], pair["pair_moments"], ranks["rank_heads"]


def recordings(d, bank, menu, count, rec):
    cv, ols = bank[:2]
    for name in np.unique(rec):
        at = np.flatnonzero(d["rec"] == name)
        on = rec == name
        ids = np.flatnonzero(on)
        q = d["query"][at]-ids[0]
        np.testing.assert_array_equal(np.unique(d["query"][at]), ids)
        cap = np.minimum(2, np.bincount(q)-1)
        eligible = menu["legal"][on, 1]
        use = np.arange(2) < count[on, None]
        initial = np.searchsorted(at, menu["row"][on][use])
        if not len(initial):
            continue
        amp = np.array([.25, .5, 1.])[menu["arm"][on][use]]
        rows, slots, _, paths, proxy, old, release, shift = timing.local_plan(
            cv[at], ols[at], d["h"][at], q, eligible, cap, initial, amp)
        on = (shift >= 0) & (shift <= model.AGE)
        idx = np.flatnonzero(on)
        rows, slots, paths, proxy, old = rows[on], slots[on], paths[on], proxy[on], np.searchsorted(idx, old)
        x, _, _, ref, scale = timing.prior.model.features(d["h"][at[initial]], amp)
        np.testing.assert_array_equal(ref, paths[old])
        yield at, x, scale, paths, (q[rows], rows, slots, proxy, release, old, eligible, cap)


def contrast(cv, y, valid, rows, paths, risk, before, after):
    """Measure the complete counterfactual but only materialize changed rows."""
    left = {rows[i]: i for i in before}
    right = {rows[i]: i for i in after}
    changed = sorted(r for r in left.keys() | right.keys() if left.get(r) != right.get(r))
    if not changed:
        return np.zeros(4)
    u = np.asarray([paths[left[r]] if r in left else cv[r] for r in changed])
    v = np.asarray([paths[right[r]] if r in right else cv[r] for r in changed])
    delta = prior.paired(u, v, y[changed], valid[changed])[1].sum()/cv.shape[1]
    return np.r_[delta, risk[after].sum(0)-risk[before].sum(0)]


def training(d, bank, menu, count, rec, cut, seed):
    rng = np.random.default_rng(seed+2803)
    xs, ys, stats = [], [], Counter()
    for at, x, scale, paths, args in recordings(d, bank, menu, count, rec):
        q, rows, slots, proxy, release, old, _, _ = args
        cv, y, valid = bank[0][at], d["y"][at], d["valid"][at]
        z = prior.moments(cv[rows], paths, y[rows], valid[rows], cut)
        risk = np.column_stack([z[k] for k in ("all_excess", "easy_excess", "easy_signed")])
        local = model.local_wait(q, slots, proxy, release, old)
        local_target = np.stack([contrast(cv, y, valid, rows, paths, risk, old[i:i+1], v[v >= 0])
                                 for i, v in enumerate(local[:, None])])
        for flags in (np.zeros(len(old), dtype=bool), proxy[old] < 0., rng.random(len(old)) < .25):
            ref, state, _ = model.walk(*args, lambda a, _: flags[a])
            target = []
            for i in range(len(old)):
                flip = flags.copy()
                flip[i] = not flip[i]
                alt, alt_state, _ = model.walk(*args, lambda a, _: flip[a])
                np.testing.assert_array_equal(state[i], alt_state[i])
                keep, wait = (alt, ref) if flags[i] else (ref, alt)
                target.append(contrast(cv, y, valid, rows, paths, risk, keep, wait))
                touched = set(slots[keep])-set(slots[wait])
                stats["other_slot_lost"] += len(touched-{i})
            target = np.stack(target)
            stats.update(labels=len(old), gain_sign_changed=int(np.sum((target[:, 0] > 0.) != (local_target[:, 0] > 0.))),
                ledger_positive=int((target[:, 0] > 0.).sum()), local_positive=int((local_target[:, 0] > 0.).sum()),
                actual_target_changed=int((np.abs(target-local_target).max(1) > 1e-10).sum()))
            xs.append(np.c_[x, state])
            ys.append((np.stack((local_target, target), 1)/scale[:, None, None]).clip(-8., 8.).astype(np.float32))
    if not xs:
        return None, stats
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))), stats


def frozen_count(d, b, rec, start, rank_meta, count_meta, donor=False):
    rank = study.load(rank_meta, lambda: study.prior.model.SwapValue(False), study.RANK_STEPS if donor else study.prior.STEPS)
    m = study.menus(d, b, rank)
    x, _, _ = base.features(d, b, m)
    width = x["motion"].shape[1]
    if donor:
        factor = donor_run.prior.model
        net = load_pair(count_meta, lambda: factor.Factorized(width))
        with torch.inference_mode():
            p = factor.moments_for_score(net(torch.from_numpy(x["motion"])))
    else:
        net = study.load(count_meta, lambda: base.model.Value(width), base.STEPS)
        with torch.inference_mode():
            p = net(torch.from_numpy(x["motion"])).numpy()
    return m, study.model.streaming_count(base.model.score(p, risk=False), m["legal"], rec, start)


@torch.inference_mode()
def evaluate(d, b, m, count, rec, cuts, old_head, heads):
    cv, ols = b[:2]
    pred = {k: cv.copy() for k in ARMS}
    acts = {k: Counter() for k in ARMS[1:]}
    for at, x, _, paths, args in recordings(d, b, m, count, rec):
        q, rows, slots, proxy, release, old, eligible, cap = args
        pred["anchor"][at[rows[old]]] = paths[old]
        v = old_head(torch.from_numpy(x)).numpy()
        flag = (v[:, 0] > 0.) & (v[:, 1:] <= 0.).all(1)
        for k in ARMS[1:]:
            if k == "old_risk":
                def decide(a, _): return flag[a]
            else:
                name, gate = k.split("_")

                def decide(a, state):
                    z = heads[name](torch.from_numpy(np.c_[x[a], state])).numpy()
                    return (z[:, 0] > 0.) & ((z[:, 1:] <= 0.).all(1) if gate == "risk" else True)
            pick, _, defer = model.walk(*args, decide)
            n = np.bincount(q[pick], minlength=len(cap))
            if (not np.isin(old[~defer], pick).all() or len(np.unique(slots[pick])) != len(pick)
                    or (n > cap).any() or not timing.capacity.respects_tokens(n, eligible)):
                raise ValueError("Ledger deployment changed an untouched edit or exceeded its budget")
            pred[k][at[rows[pick]]] = paths[pick]
            acts[k].update(deferred=int(defer.sum()), moved=int((q[pick] > release[slots[pick]]).sum()), dropped=len(old)-len(pick))
    metrics = {k: parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "anchor") for k in ARMS[1:]] + [(f"ledger_{g}", f"local_{g}") for g in ("gain", "risk")]
    direct = {f"{a}_vs_{b}": float(100.*prior.paired(pred[b], pred[a], y, valid)[1].sum()/mass) for a, b in pairs}
    return dict(metrics=metrics, direct=direct, actions=acts)


def execute(key, a, ph, h, previous, prev, old_heads, pair, ranks, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    context = {}
    for s in sites:
        f = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(f, allow_pickle=False) as z:
            context[s] = z["recording"], z["start"]
    rows, frozen, updates = {}, {}, 0
    def cuts(keep):
        return np.quantile(np.concatenate([sites[s]["cut_samples"] for s in keep]), [.25, .75])

    @lru_cache(maxsize=12)
    def donor_rows(held, site, seed):
        omit = sorted((held, site))
        fit = [s for s in sites if s not in omit]
        tail = "_".join(omit)+f"_{seed}"
        rk, cn = ranks["rank_"+tail], pair["moments_"+tail]
        for m in (rk, cn):
            if m["identity"]["fit"] != fit or m["identity"]["seed"] != seed:
                raise ValueError("Training-time roster saw a held site")
        d, b = sites[site], banks[site]
        rec, start = context[site]
        m, count = frozen_count(d, b, rec, start, rk, cn, donor=True)
        return training(d, b, m, count, rec, cuts(fit)[0], seed)

    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            data, label_stats = [], Counter()
            for site in donors:
                train, stats = donor_rows(held, site, seed)
                if train is not None:
                    data.append(train)
                label_stats.update(stats)
            batch = timing.prior.batcher(data)
            width = data[0][0].shape[1]
            heads = {}
            for i, mode in enumerate(MODES):
                identity = dict(registration=key, held=held, donors=donors, seed=seed, mode=mode)
                heads[mode], frozen[f"{name}_{mode}"], u = study.fit(identity, f"{name}_{mode}", seed,
                    lambda: prior.model.WaitValue(width), STEPS, batch,
                    lambda net, x, z: prior.model.loss(net, x, z[:, i]), replay, beat)
                updates += u
            rec, start = context[held]
            b = banks[held]
            m, count = frozen_count(d, b, rec, start, ph["heads"][f"{held}/{seed}/additive_rank"], h["heads"][name+"_motion"])
            old_head = study.load(old_heads[name], lambda: prior.model.WaitValue(width-5), prior.STEPS)
            v = evaluate(d, b, m, count, rec, cuts(donors), old_head, heads)
            old = next(r for r in prev["by_site"][held] if r["seed"] == seed)
            for k, ref in (("anchor", "anchor"), ("old_risk", "predicted_gain_risk")):
                for field, x in v["metrics"][k].items():
                    np.testing.assert_allclose(x, old["metrics"][ref][field], atol=1e-10, rtol=0.)
            rows[held].append(dict(seed=seed, label_stats=label_stats, **v))
            beat(phase="held", held=held, seed=seed, labels=label_stats["labels"])
    flat = [r for v in rows.values() for r in v]
    draw = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    summary = dict(arms={k: dict(vs_cv=ci(lambda r: r["metrics"][k]["vs_cv"]),
        **{f: sum(r["metrics"][k][f] for r in flat) for f in fields}) for k in ARMS},
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]},
        label_stats=sum((Counter(r["label_stats"]) for r in flat), Counter()),
        actions={k: sum((Counter(r["actions"][k]) for r in flat), Counter()) for k in ARMS[1:]})
    receipt = OUT / "head_freeze.json"
    run.once(receipt, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, heads=run.sha(receipt),
        by_site=rows, original_2pct=True, confirmation_access=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            event = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", event)
            if v["phase"] != "fit":
                print(event, flush=True)

        updates = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=updates)


if __name__ == "__main__":
    main()
