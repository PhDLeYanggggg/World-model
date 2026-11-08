"""Test whether past-only values and neighbor context predict useful intervention."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.stats import rankdata

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_token_count_teacher as prior
from scripts.audit_m3w_recording_count_matched import exact_score_plan
from src.world_model import m3w_direct_count_value as model

study, parent, run, torch = prior.study, prior.parent, prior.run, prior.torch
OUT = prior.OUT.parent / "direct_count_value_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
MODES = ("motion", "peers")
ARMS = ("past_proxy", "motion_gain_only", "motion", "peers_gain_only", "peers")
STEPS = 700


def register():
    key, a, ph, h = prior.register()
    files = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze")]
    p, f = [run.read(path) for path in files]
    if any(v["registration"] != key for v in (p, f)) or p["heads"] != run.sha(files[1]) or p["confirmation_access"]:
        raise ValueError("Require bound chronological teacher results")
    path = OUT / "registration.json"
    run.once(path, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in files], matching=run.artifact(Path(exact_score_plan.__code__.co_filename)),
        fixed_modes=MODES, steps=STEPS, seeds=study.prior.SEEDS, native_steps=[8, 12], workers=0, threads=4,
        primary="peers vs motion under identical causal credit rule; same-total retrospective attribution",
        cheap_control="observed prefix interpolation errors only", loss="six signed moments and four outcome events",
        score="min conservative gain minus fixed positive all/easy risk excess and net easy penalties, easy support bonus",
        no_threshold_tuning=True, original_2pct=True, confirmation_access=False)))
    return run.sha(path), a, ph, h


def features(d, bank, menu):
    g = list(study.prior.groups(d))
    context = model.contexts(d["h"], menu["row"], g)
    scale = np.array([np.median(bank[4][ix]) for ix, _ in g])
    proxy = model.past_proxy(d["h"], menu["row"], menu["arm"], scale)
    x = np.c_[menu["x"], proxy[:, 1:]].astype(np.float32)
    return dict(motion=np.c_[x, np.zeros_like(context)], peers=np.c_[x, context]), scale, proxy


def labels(d, bank, menu, cut, scale):
    return model.targets(*bank[:3], d["y"], d["valid"], cut, menu["row"], menu["arm"], scale)


def auc(v, label):
    label = np.asarray(label, dtype=bool)
    n, m = label.sum(), (~label).sum()
    if not n or not m:
        return None
    return float((rankdata(v)[label].sum()-n*(n+1)/2)/(n*m))


def discrimination(pred, y, legal):
    on = legal.copy()
    on[:, 0] = False
    z, t = pred[on], y[on]
    return dict(actions=len(z), gain_auc=auc(z[:, 6], t[:, 6]),
                all_excess_auc=auc(z[:, 7], t[:, 7]), easy_excess_auc=auc(z[:, 8], t[:, 8]),
                easy_support_auc=auc(z[:, 9], t[:, 9]),
                gain_mae=float(np.abs(z[:, 0]-t[:, 0]).mean()),
                easy_excess_mae=float(np.abs(z[:, 3]-t[:, 3]).mean()))


def batches(data):
    tensors = [tuple(torch.from_numpy(v) for v in (m["x"], m["legal"], y)) for m, y in data if len(y)]

    def sample(rng):
        g = rng.integers(len(tensors), size=128)
        indices = [rng.integers(len(v[0]), size=(g == i).sum()) for i, v in enumerate(tensors)]
        return tuple(torch.cat([v[j][i] for v, i in zip(tensors, indices, strict=True)]) for j in range(3))

    return sample


def match(rec, legal, starts, scores, online):
    exact = {k: np.zeros(len(rec), dtype=int) for k in scores}
    nonzero = 0
    for name in np.unique(rec):
        ix = rec == name
        total = min(int(v[ix].sum()) for v in online.values())
        if not total:
            continue
        nonzero += 1
        for k, s in scores.items():
            plan = exact_score_plan(s[ix], legal[ix], total)
            proxy = np.full_like(s[ix], -1.)
            proxy[np.arange(len(plan)), plan] = 1.
            np.testing.assert_array_equal(study.model.streaming_count(proxy, legal[ix], rec[ix], starts[ix]), plan)
            exact[k][ix] = plan
    return exact, nonzero


@torch.inference_mode()
def evaluate(d, bank, menu, x, proxy, y, nets, cuts):
    cv, ols, b = bank[:3]
    rec, starts, legal = d["query_rec"], d["starts"], menu["legal"]
    pred = {k: net(torch.from_numpy(x[k])).numpy() for k, net in nets.items()}
    if any(not np.isfinite(v).all() for v in pred.values()):
        raise ValueError("Nonfinite conditional moment")
    scores = {"past_proxy": proxy}
    for k, v in pred.items():
        scores[k] = model.score(v)
        scores[k+"_gain_only"] = model.score(v, risk=False)
    counts = {k: study.model.streaming_count(s, legal, rec, starts) for k, s in scores.items()}
    matched, nonzero = match(rec, legal, starts, {k: scores[k] for k in MODES}, {k: counts[k] for k in MODES})

    def score(counts0):
        paths, metrics = {}, {}
        for name, c in counts0.items():
            use = np.arange(2) < c[:, None]
            at = np.full(len(cv), -1, dtype=int)
            at[menu["row"][use]] = menu["arm"][use]
            np.testing.assert_array_equal(np.bincount(d["query"][at >= 0], minlength=len(c)), c)
            paths[name] = parent.selected(cv, b, at)
            metrics[name] = parent.score(d, cv, ols, paths[name], cuts)
        return paths, metrics

    paths, observed = score(counts)
    mp, mm = score(matched)
    target, mask = d["y"], d["valid"]
    mass = np.linalg.norm(cv-target, axis=-1)[mask].sum()

    def contrast(path, a, b):
        return float(100.*parent.transfer.paired(path[b], path[a], target, mask)[1].sum()/mass)

    return dict(online=observed, retrospective_matched=mm, nonzero_matched_recordings=nonzero,
        discrimination={k: discrimination(v, y, legal) for k, v in pred.items()},
        direct=dict(peers_vs_motion=contrast(paths, "peers", "motion"), peers_vs_proxy=contrast(paths, "peers", "past_proxy"),
                    peers_vs_gain_only=contrast(paths, "peers", "peers_gain_only"),
                    peers_vs_motion_matched=contrast(mp, "peers", "motion")))


def execute(key, a, ph, h, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        path = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(path, allow_pickle=False) as z:
            d.update(query_rec=z["recording"], starts=z["start"])
    bank = {s: parent.prior.analytic(d) for s, d in sites.items()}
    rank_builder = lambda: study.prior.model.SwapValue(False)
    heads, rows, steps = {}, {}, 0
    for held, d in sites.items():
        rows[held] = []
        donors = [s for s in sites if s != held]
        cuts = lambda ss: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in ss]), [.25, .75])
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            training = {k: [] for k in MODES}
            for donor in donors:
                omit = sorted((held, donor))
                name = "rank_"+"_".join(omit)+f"_{seed}"
                meta = h["rank_heads"][name]
                if set(meta["identity"]["fit"]) != set(sites)-set(omit):
                    raise ValueError("Donor ranker violated the fit split")
                rank = study.load(meta, rank_builder, study.RANK_STEPS)
                menu = study.menus(sites[donor], bank[donor], rank)
                x, scale, _ = features(sites[donor], bank[donor], menu)
                y = labels(sites[donor], bank[donor], menu, cuts([s for s in sites if s not in omit])[0], scale)
                on = menu["legal"][:, 1]
                for k in MODES:
                    training[k].append((dict(x=x[k][on], legal=menu["legal"][on]), y[on]))
            nets = {}
            for k in MODES:
                name = f"{held}_{seed}_{k}"
                identity = dict(registration=key, held=held, donors=donors, seed=seed, mode=k)
                nets[k], heads[name], n = study.fit(identity, name, seed, lambda: model.Value(training[k][0][0]["x"].shape[-1]),
                    STEPS, batches(training[k]), lambda net, x, legal, y: model.objective(net(x), y, legal), replay, beat)
                steps += n
            rank = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], rank_builder, study.prior.STEPS)
            menu = study.menus(d, bank[held], rank)
            x, scale, proxy = features(d, bank[held], menu)
            cut = cuts(donors)
            y = labels(d, bank[held], menu, cut[0], scale)
            rows[held].append(dict(seed=seed, **evaluate(d, bank[held], menu, x, proxy, y, nets, cut)))
            beat(phase="held", held=held, seed=seed)
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    flat = [r for v in rows.values() for r in v]

    def ci(fn):
        vals = [[fn(r) for r in v] for v in rows.values()]
        if any(y is None for x in vals for y in x):
            return {"point": None, "interval95": None, "defined_location_seeds": sum(y is not None for x in vals for y in x)}
        x = np.asarray(vals).mean(1)
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    def collect(family, keys):
        return {k: dict(vs_cv=ci(lambda r: r[family][k]["vs_cv"]), vs_ols4=ci(lambda r: r[family][k]["vs_ols4"]),
            **{f: sum(r[family][k][f] for r in flat) for f in ("edits", "dual_positive", "risk_pass", "undefined", "views")}) for k in keys}

    summary = dict(online=collect("online", ARMS), matched=collect("retrospective_matched", MODES),
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]},
        discrimination={k: {f: ci(lambda r: r["discrimination"][k][f]) for f in flat[0]["discrimination"][k] if f != "actions"} for k in MODES},
        nonzero_matched_recordings=sum(r["nonzero_matched_recordings"] for r in flat))
    path = OUT / "head_freeze.json"
    run.once(path, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(path), summary=summary, by_site=rows,
        exact_totals_retrospective_not_online=True, matched_allocation_no_labels=True, original_2pct=True, confirmation_access=False)))
    return steps


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
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] != "fit":
                print(row, flush=True)

        n = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=n, replay=args.replay)


if __name__ == "__main__":
    main()
