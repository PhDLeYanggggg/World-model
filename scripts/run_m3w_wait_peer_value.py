"""Train and test whether aligned peer history identifies useful actual wait moves."""

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
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_ledger_wait as prior
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_wait_peer_value as model

run, torch, parent, study = prior.run, prior.torch, prior.parent, prior.study
OUT, PRIVATE = prior.OUT.parent / "wait_peer_value_v1", prior.PRIVATE.parent / "wait_peer_value_v1"
STEPS = 800


def register():
    key, a, ph, h, _, _, _, pair, ranks = prior.register()
    p, f, c = (prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze", "capacity/analysis"))
    old, heads, cap = (run.read(k) for k in (p, f, c))
    if (old["registration"] != key or heads["registration"] != key or old["heads"] != run.sha(f)
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (old, cap))):
        raise ValueError("Require frozen whole-ledger experiment")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(k) for k in (p, f, c)], steps=STEPS, seeds=study.prior.SEEDS, native_steps=[8, 12],
        arms=model.MODES, features="same observed ego/ledger/opportunity history, zero/aligned/wrong-focal observed peers",
        labels="actual one-toggle outcome: moved, positive gain, moved and positive and joint supported, paired gain",
        inference="first released query with p_joint >= .25, p_moved >= .5, estimated gain > 0; at most one toggle",
        matching="evaluate full policy; for utility alone replay original slots matching actually emitted slot identities",
        fitting="pair-excluded nominations and count heads from parent; unchanged train/held sites",
        original_2pct=True, confirmation_access=False, no_held_tuning=True)))
    return run.sha(reg), a, ph, h, pair, ranks, old, cap


def packets(d, b, menu, count, rec):
    peak = prior.timing.model.shape.peak(b[1]-b[0])
    for at, x, scale, paths, args in prior.recordings(d, b, menu, count, rec):
        q, rows, slots, proxy, release, old, eligible, cap = args
        query = d["query"][at]-d["query"][at[0]]
        _, state, _ = prior.model.walk(*args, lambda a, _: np.zeros(len(a), dtype=bool))
        actual, wrong = model.peer_features(d["h"][at], query, rows[old])
        context = model.opportunity_features(query, peak[at] > 0., peak[at], cap, release)
        common = np.c_[x, state, context]
        z = np.stack([np.c_[common, p] for p in (np.zeros_like(actual), actual, wrong)], axis=1)
        yield at, z, scale, paths, args


def prediction(cv, rows, paths, chosen):
    pred = cv.copy()
    pred[rows[chosen]] = paths[chosen]
    return pred


def read_joint(cv, ols, path, y, valid, cuts):
    a = policy(path, cv, cv, y, valid, np.zeros(len(cv)), easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]["0.0"]
    return a["supported"] and all(prior.prior.paired(ref, path, y, valid)[1].sum() > 0. for ref in (cv, ols))


def labels(cv, ols, y, valid, scale, paths, args, cuts):
    _, rows, slots, _, _, old, _, _ = args
    anchor = prediction(cv, rows, paths, old)
    actual, target = [], []
    for i in range(len(old)):
        pick, _, _ = prior.model.walk(*args, lambda a, _: a == i)
        if not np.isin(np.delete(old, i), pick).all():
            raise ValueError("A single wait changed another anchor edit")
        moved = len(pick) == len(old)
        path = prediction(cv, rows, paths, pick)
        delta = prior.prior.paired(anchor, path, y, valid)[1].sum()
        good = delta > 0.
        joint = moved and good and read_joint(cv, ols, path, y, valid, cuts)
        target.append([moved, good, joint, np.clip(delta/(cv.shape[1]*scale[i]), -8., 8.)])
        actual.append(pick)
    return np.asarray(target, dtype=np.float32), actual


def training(d, b, menu, count, rec, cuts):
    x, y = [], []
    for at, z, scale, paths, args in packets(d, b, menu, count, rec):
        target, _ = labels(b[0][at], b[1][at], d["y"][at], d["valid"][at], scale, paths, args, cuts)
        x.append(z)
        y.append(target)
    return tuple(torch.from_numpy(np.concatenate(v)) for v in (x, y)) if x else None


def skill(t, p):
    if not len(t):
        return dict(n=0, positive=0, auroc=None, ap=None, brier=None)
    positive = int(t.sum())
    defined = 0 < positive < len(t)
    return dict(n=len(t), positive=positive, auroc=float(roc_auc_score(t, p)) if defined else None,
                ap=float(average_precision_score(t, p)) if defined else None, brier=float(np.square(t-p).mean()))


@torch.inference_mode()
def evaluate(d, b, menu, count, rec, heads, cuts):
    cv, ols = b[:2]
    y, valid = d["y"], d["valid"]
    pred = {k: cv.copy() for k in ("anchor", *model.MODES)}
    match = {k: cv.copy() for k in model.MODES}
    raw, truth, stats = {k: [] for k in model.MODES}, [], {k: Counter() for k in model.MODES}
    for at, z, scale, paths, args in packets(d, b, menu, count, rec):
        q, rows, slots, _, release, old, eligible, cap = args
        pred["anchor"][at[rows[old]]] = paths[old]
        for i, k in enumerate(model.MODES):
            value = heads[k](torch.from_numpy(z[:, i].copy())).numpy()
            probability = expit(value[:, :3])
            choice = model.causal_pick(release, probability[:, 2], probability[:, 0], value[:, 3])
            pick, _, _ = prior.model.walk(*args, lambda a, _: a == choice if choice is not None else np.zeros(len(a), bool))
            counts = np.bincount(q[pick], minlength=len(cap))
            if ((counts > cap).any() or len(np.unique(slots[pick])) != len(pick)
                    or not prior.timing.capacity.respects_tokens(counts, eligible)):
                raise ValueError("Policy exceeds causal credit")
            pred[k][at[rows[pick]]] = paths[pick]
            kept = old[slots[pick]]
            match[k][at[rows[kept]]] = paths[kept]
            raw[k].append(probability)
            stats[k].update(accepted=int(choice is not None), moved=int((q[pick] > release[slots[pick]]).sum()),
                            expired=len(old)-len(pick))
        target, _ = labels(cv[at], ols[at], y[at], valid[at], scale, paths, args, cuts)
        truth.append(target)
    truth = np.concatenate(truth) if truth else np.empty((0, 4))
    metrics = {k: parent.score(d, cv, ols, v, cuts) for k, v in pred.items()}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pair = lambda a, b: float(100.*prior.prior.paired(b, a, y, valid)[1].sum()/mass)
    direct = {f"{k}_vs_anchor": pair(pred[k], pred["anchor"]) for k in model.MODES}
    direct.update({f"{k}_vs_emission_match": pair(pred[k], match[k]) for k in model.MODES})
    direct.update(peer_vs_self=pair(pred["peer"], pred["self"]), peer_vs_wrong=pair(pred["peer"], pred["wrong_peer"]))
    scores = {}
    for k, p in raw.items():
        v = np.concatenate(p) if p else np.empty((0, 3))
        scores[k] = dict(moved=skill(truth[:, 0], v[:, 0]), positive=skill(truth[:, 1], v[:, 1]),
                         joint_positive=skill(truth[:, 2], v[:, 2]),
                         positive_given_actual_move=skill(truth[truth[:, 0] > 0., 1], v[truth[:, 0] > 0., 1]))
    return dict(metrics=metrics, direct=direct, decisions=stats, probe=scores)


def execute(key, a, ph, h, pair, ranks, old, cap, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    context = {}
    for s in sites:
        f = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(f, allow_pickle=False) as z:
            context[s] = z["recording"], z["start"]
    rows, frozen, steps = {}, {}, 0

    def cuts(fit):
        return np.quantile(np.concatenate([sites[s]["cut_samples"] for s in fit]), [.25, .75])

    @lru_cache(maxsize=10)
    def donor_data(held, site, seed):
        omit = sorted((held, site))
        fit = [s for s in sites if s not in omit]
        tail = "_".join(omit)+f"_{seed}"
        rank, counter = ranks["rank_"+tail], pair["moments_"+tail]
        for z in (rank, counter):
            if z["identity"]["fit"] != fit or z["identity"]["seed"] != seed:
                raise ValueError("Donor feature producer has held-site exposure")
        d, b = sites[site], banks[site]
        rec, start = context[site]
        m, count = prior.frozen_count(d, b, rec, start, rank, counter, donor=True)
        return training(d, b, m, count, rec, cuts(fit))

    for held, d in sites.items():
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            donors = [s for s in sites if s != held]
            data = [x for s in donors if (x := donor_data(held, s, seed)) is not None]
            width = data[0][0].shape[-1]
            batch = prior.timing.prior.batcher(data)
            heads = {}
            for i, mode in enumerate(model.MODES):
                ident = dict(registration=key, held=held, donors=donors, seed=seed, mode=mode)
                heads[mode], frozen[f"{name}_{mode}"], u = study.fit(ident, f"{name}_{mode}", seed,
                    lambda: model.Value(width), STEPS, batch, lambda net, x, y: model.loss(net, x[:, i], y), replay, beat)
                steps += u
            rec, start = context[held]
            b = banks[held]
            m, count = prior.frozen_count(d, b, rec, start, ph["heads"][f"{held}/{seed}/additive_rank"], h["heads"][name+"_motion"])
            v = evaluate(d, b, m, count, rec, heads, cuts(donors))
            ref = next(r for r in old["by_site"][held] if r["seed"] == seed)
            diag = next(r for r in cap["by_site"][held] if r["seed"] == seed)["actual_decoder"]
            for k, y in v["metrics"]["anchor"].items():
                np.testing.assert_allclose(y, ref["metrics"]["anchor"][k], atol=1e-10, rtol=0.)
            if v["probe"]["self"]["joint_positive"]["positive"] != diag.get("improves_anchor_and_joint_matched", 0):
                raise ValueError("Exact joint-positive opportunity labels did not reproduce")
            rows[held].append(dict(seed=seed, **v))
            beat(phase="held", held=held, seed=seed, opportunities=v["probe"]["self"]["joint_positive"])
    flat = [r for v in rows.values() for r in v]

    def ci(fn):
        means = []
        for v in rows.values():
            x = [z for r in v if (z := fn(r)) is not None]
            if x:
                means.append(np.mean(x))
        x = np.array(means)
        if not len(x):
            return dict(point=None, interval95=None, locations=0)
        draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(), locations=len(x))

    def contrast(r, ref):
        a, b = (r["probe"][k]["joint_positive"]["ap"] for k in ("peer", ref))
        return None if a is None or b is None else a-b

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    summary = dict(arms={k: {f: sum(r["metrics"][k][f] for r in flat) for f in fields} for k in flat[0]["metrics"]},
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]},
        probe={k: {p: {f: ci(lambda r: r["probe"][k][p][f]) for f in ("ap", "auroc", "brier")}
               for p in flat[0]["probe"][k]} for k in model.MODES},
        joint_ap_contrast={k: ci(lambda r: contrast(r, k)) for k in ("self", "wrong_peer")},
        opportunities={f: sum(r["probe"]["self"]["joint_positive"][f] for r in flat) for f in ("n", "positive")},
        actions={k: sum((Counter(r["decisions"][k]) for r in flat), Counter()) for k in model.MODES})
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, heads=run.sha(f),
        by_site=rows, original_2pct=True, confirmation_access=False)))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
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

        updates = execute(*register(), a.replay, beat)
        beat(phase="complete", updates=updates)


if __name__ == "__main__":
    main()
