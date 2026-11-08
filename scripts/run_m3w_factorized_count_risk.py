"""Test a factorized risk head without changing nominations, scoring or risk rules."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_support_value_attribution as attribution
from src.world_model import m3w_factorized_count_risk as model

study, parent, run, torch = attribution.study, attribution.parent, attribution.run, attribution.torch
OUT = attribution.OUT.parent / "factorized_count_risk_v1"
PRIVATE = study.PRIVATE.parent / OUT.name
ARMS = ("gain_only", "signed", "factorized_risk", "factorized_gain", "factorized_both")


def register():
    key, a, ph, p, h = attribution.register()
    source = attribution.OUT / "analysis.json"
    diagnostic = run.read(source)
    if (diagnostic["registration"] != key or diagnostic["confirmation_access"] or not diagnostic["original_2pct"]
            or not diagnostic["labels_used_only_for_oracles_and_evaluation"]):
        raise ValueError("Require the bound support attribution")
    path = OUT / "registration.json"
    run.once(path, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        attribution=run.artifact(source), arms=ARMS, native_steps=[8, 12], updates=study.STEPS, seeds=study.study.prior.SEEDS,
        primary="replace signed risk only; preserve gains, menus, scoring weights and tokens",
        matching="exact same recording totals as motion gain-only; retrospective, label blind allocation",
        no_threshold_tuning=True, original_2pct=True, confirmation_access=False)))
    return run.sha(path), a, ph, p, h


@torch.inference_mode()
def evaluate(d, bank, menu, frozen, new, x, truth, cuts):
    cv, ols, b = bank[:3]
    rec, starts, legal = d["query_rec"], d["starts"], menu["legal"]
    t = torch.from_numpy(x)
    a = frozen(t).numpy()
    z = model.moments_for_score(new(t))
    r, g = a.copy(), a.copy()
    r[..., [2, 3, 4, 5, 9]] = z[..., [2, 3, 4, 5, 9]]
    g[..., :2] = z[..., :2]
    scorer = study.model.score
    s = dict(gain_only=scorer(a, risk=False), signed=scorer(a), factorized_risk=scorer(r),
             factorized_gain=scorer(g), factorized_both=scorer(z))
    on = {k: study.study.model.streaming_count(v, legal, rec, starts) for k, v in s.items()}
    matched = {k: np.zeros(len(rec), dtype=int) for k in ARMS}
    for name in np.unique(rec):
        ix = rec == name
        n = int(on["gain_only"][ix].sum())
        for k, v in s.items():
            if n:
                matched[k][ix] = study.exact_score_plan(v[ix], legal[ix], n)
                if not attribution.method.respects_tokens(matched[k][ix], legal[ix, 1]):
                    raise ValueError("Fixed totals must respect chronological credit")
    mass = np.linalg.norm(cv-d["y"], axis=-1)[d["valid"]].sum()

    def collect(c):
        paths = {k: attribution.install(cv, b, menu, v) for k, v in c.items()}
        return dict(scores={k: parent.score(d, cv, ols, p, cuts) for k, p in paths.items()},
                    direct={k: float(100.*parent.transfer.paired(paths["signed"], paths[k], d["y"], d["valid"])[1].sum()/mass)
                            for k in ARMS if k != "signed"})

    mask = legal.copy()
    mask[:, 0] = False
    quality = {k: {"risk_mae": np.abs((v-truth)[mask][:, [2, 3]]).mean(0).tolist(),
                   "gain_mae": float(np.abs((v-truth)[mask][:, 0]).mean())}
               for k, v in (("signed", a), ("factorized", z))}
    return dict(online=collect(on), matched=collect(matched), quality=quality)


def execute(key, a, ph, p, h, replay, beat):
    _, _, _, donor_heads = study.prior.register()
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        path = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(path, allow_pickle=False) as z:
            d.update(query_rec=z["recording"], starts=z["start"])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    rank_builder = lambda: study.study.prior.model.SwapValue(False)
    rows, heads, steps = {}, {}, 0
    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        cut = lambda ss: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in ss]), [.25, .75])
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            identity = dict(registration=key, held=held, donors=donors, seed=seed)
            train = []
            for donor in donors:
                omit = sorted((held, donor))
                rank = donor_heads["rank_heads"]["rank_"+"_".join(omit)+f"_{seed}"]
                fit = [s for s in sites if s not in omit]
                if rank["identity"]["fit"] != fit:
                    raise ValueError("Donor ranker saw a held source")
                menu = study.study.menus(sites[donor], banks[donor], study.study.load(rank, rank_builder, study.study.RANK_STEPS))
                x, scale, _ = study.features(sites[donor], banks[donor], menu)
                src = sites[donor]
                y = model.targets(*banks[donor][:3], src["y"], src["valid"], cut(fit)[0], menu["row"], menu["arm"], scale)
                on = menu["legal"][:, 1]
                train.append((dict(x=x["motion"][on], legal=menu["legal"][on]), y[on]))
            width = train[0][0]["x"].shape[1]
            net, heads[name], n = study.study.fit(identity, name, seed, lambda: model.Factorized(width), study.STEPS,
                study.batches(train), lambda net, x, legal, y: model.objective(net(x), y, legal), replay, beat)
            steps += n
            rank = study.study.load(ph["heads"][f"{held}/{seed}/additive_rank"], rank_builder, study.study.prior.STEPS)
            menu = study.study.menus(d, banks[held], rank)
            x, scale, _ = study.features(d, banks[held], menu)
            frozen = study.study.load(h["heads"][name+"_motion"], lambda: study.model.Value(width), study.STEPS)
            cuts = cut(donors)
            y = study.labels(d, banks[held], menu, cuts[0], scale)
            z = evaluate(d, banks[held], menu, frozen, net, x["motion"], y, cuts)
            ref = next(v for v in p["by_site"][held] if v["seed"] == seed)
            for k, pkey in (("gain_only", "motion_gain_only"), ("signed", "motion")):
                if run.stable(z["online"]["scores"][k]) != run.stable(ref["online"][pkey]):
                    raise ValueError("Frozen policy changed")
            rows[held].append(dict(seed=seed, **z))
            beat(phase="held", held=held, seed=seed)
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    flat = [v for vv in rows.values() for v in vv]

    def ci(fn):
        x = np.array([np.mean([fn(v) for v in vv]) for vv in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    out = {r: {k: dict(vs_cv=ci(lambda v: v[r]["scores"][k]["vs_cv"]),
            direct_vs_signed=ci(lambda v: v[r]["direct"][k]) if k != "signed" else None,
            **{f: sum(v[r]["scores"][k][f] for v in flat) for f in ("edits", "dual_positive", "risk_pass", "undefined", "views")})
            for k in ARMS} for r in ("online", "matched")}
    out["quality"] = {k: dict(gain_mae=ci(lambda v: v["quality"][k]["gain_mae"]),
        all_risk_mae=ci(lambda v: v["quality"][k]["risk_mae"][0]), easy_risk_mae=ci(lambda v: v["quality"][k]["risk_mae"][1]))
        for k in ("signed", "factorized")}
    path = OUT / "head_freeze.json"
    run.once(path, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(path), summary=out, by_site=rows,
        original_2pct=True, confirmation_access=False, matching_label_blind_not_online=True)))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study.study.PRIVATE = PRIVATE
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
