"""Three matched training objectives on the existing OpenDD development roster."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_comoving_eqmotion as base
from src.world_model import m3w_completion_forecast as loss

run, parent, np, torch = base.run, base.parent, base.np, base.torch
OUT = base.OUT.parent / "opendd_completion_forecast_v1"
PRIVATE = base.PRIVATE.parent / OUT.name
ARMS = loss.OBJECTIVES
SEEDS, STEPS, BATCH = (17, 29, 43), 800, 32


def register():
    key, roster, credit, rank, matched, source = base.register()
    p = base.OUT / "analysis.json"
    a = run.read(p)
    if a["registration"] != key or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Need verified OpenDD-only predictor and roster")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(
        code=run.artifact(Path(__file__)), loss=run.artifact(Path(loss.__file__)),
        parent=run.artifact(p), trainer=run.artifact(Path(base.two.fit.__file__)),
        architecture=base.model.ARCH, predictor="comoving_ego",
        arms=ARMS, seeds=SEEDS, steps=STEPS, batch=BATCH, threads=4, workers=0,
        protocol="native obs8/pred12; all past-complete rows, including partial or absent futures",
        sampling="same site-balanced train draws and init; held site excluded; scale from history",
        primary="direct dual vs masked conservative full-grid gain plus absolute CV and OLS4 gain",
        secondary="CV debit only, observed accuracy and edit-radius shrinkage; same old fixed slots and peaks",
        uncertainty="3000 paired site bootstrap draws after averaging seeds",
        no_held_selection=True, original_2pct=True, confirmation_access=False)))
    return run.sha(f), roster, credit, rank, matched, source


def batches(rows):
    def sample(rng):
        sizes = rng.multinomial(BATCH, np.full(len(rows), 1. / len(rows)))
        ids = [rng.integers(len(t[0]), size=n) for t, n in zip(rows, sizes, strict=True)]
        return tuple(torch.cat([t[j][i] for t, i in zip(rows, ids, strict=True)])
                     for j in range(len(rows[0])))
    return sample


def attach_refs(context, y, valid, bank):
    return (*context, y, valid, *(torch.from_numpy(a.astype(np.float32)) for a in bank[:2]))


def freeze_predictions(name, paths):
    f = PRIVATE / f"{name}_paths.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as z:
            if set(z.files) != set(paths):
                raise ValueError("Unexpected prediction arms")
            for k, p in paths.items():
                np.testing.assert_array_equal(z[k], p)
    else:
        tmp = f.with_suffix(".pending.npz")
        np.savez(tmp, **paths)
        tmp.replace(f)
    return run.artifact(f)


def full_metrics(d, bank, paths, cuts):
    cv, ols = bank[:2]
    y, valid = d["y"], d["valid"]
    full, endpoint = valid.all(1), valid[:, -1]
    result = dict(complete=int(full.sum()), endpoint=int(endpoint.sum()), arms={})
    m = np.linalg.norm(cv - y, axis=-1)[valid].sum()
    for k, p in paths.items():
        err = np.linalg.norm(p - y, axis=-1)
        pairs = {r: base.paired(ref, p, y, valid) for r, ref in (("cv", cv), ("ols4", ols))}
        result["arms"][k] = dict(
            complete_ade_sum=float(err[full].sum() / y.shape[1]),
            endpoint_fde_sum=float(err[endpoint, -1].sum()),
            p95_complete_ade=float(np.quantile(err[full].mean(1), .95)),
            known={r: float(100. * v[0].sum() / m) for r, v in pairs.items()},
            lower={r: float(100. * v[1].sum() / m) for r, v in pairs.items()},
            peak_cv_mean=float(base.radius.peak(p - cv).mean()),
            record=parent.score(d, cv, ols, p, cuts))
    result["effects"] = {}
    for k in ARMS[1:]:
        known, lower = base.paired(paths["masked"], paths[k], d["y"], d["valid"])
        result["effects"][k] = dict(known=float(100. * known.sum() / m),
                                    lower=float(100. * lower.sum() / m))
    return result


def matched_metrics(d, bank, paths, packet, menu, count, cuts, reference):
    cv, ols = bank[:2]
    use = np.arange(2) < count[:, None]
    at, arm = menu["row"][use], menu["arm"][use]
    anchor = packet["paths"][at, arm, 0]
    prefix = packet["paths"][at, arm, 3]
    radii = base.radius.peak(anchor - cv[at])
    paths = {"anchor": anchor, "prefix_match": prefix, **{
        k: base.radius.match_radius(cv[at], anchor, (p[at] - cv[at])[:, None])[:, 0]
        for k, p in paths.items()}}
    result, gains = {}, {}
    mass = np.linalg.norm(cv - d["y"], axis=-1)[d["valid"]].sum()
    incumbent = cv.copy()
    incumbent[at] = anchor
    for k, p in paths.items():
        np.testing.assert_allclose(base.radius.peak(p - cv[at]), radii, atol=1e-10)
        forecast = cv.copy()
        forecast[at] = p
        np.testing.assert_array_equal(np.flatnonzero((forecast != cv).any((1, 2))), np.sort(at))
        result[k] = parent.score(d, cv, ols, forecast, cuts)
        if k in reference["policies"]:
            for metric, val in reference["policies"][k].items():
                np.testing.assert_allclose(result[k][metric], val, atol=1e-10, rtol=0.)
        gains[k] = float(100. * base.paired(incumbent, forecast, d["y"], d["valid"])[1].sum() / mass)
    return dict(policies=result, gains_vs_anchor=gains)


def summarize(by_site):
    flat = [r for rows in by_site.values() for r in rows]
    n = len(by_site)
    draw = np.random.default_rng(817).integers(n, size=(3000, n))

    def interval(get):
        site = np.array([np.mean([get(v) for v in rows]) for rows in by_site.values()])
        return dict(point=float(site.mean()), worst_site=float(site.min()),
                    positive_sites=int((site > 0.).sum()),
                    interval95=np.quantile(site[draw].mean(1), [.025, .975]).tolist())

    def aggregate(get, keys):
        return {k: sum(get(row)[k] for row in flat) for k in keys}

    keys = ("edits", "views", "undefined", "risk_pass", "dual_positive")
    raw = {}
    for k in ARMS:
        get = lambda v: v["full"]["arms"][k]
        raw[k] = dict(
            ade=sum(get(v)["complete_ade_sum"] for v in flat) / sum(v["full"]["complete"] for v in flat),
            fde=sum(get(v)["endpoint_fde_sum"] for v in flat) / sum(v["full"]["endpoint"] for v in flat),
            mean_peak_cv=float(np.mean([get(v)["peak_cv_mean"] for v in flat])),
            record=aggregate(lambda v: get(v)["record"], keys),
            lower={r: interval(lambda v: get(v)["lower"][r]) for r in ("cv", "ols4")},
            known={r: interval(lambda v: get(v)["known"][r]) for r in ("cv", "ols4")})
    return dict(raw=raw, direct_vs_masked={k: {
        m: interval(lambda v: v["full"]["effects"][k][m]) for m in ("known", "lower")}
        for k in ARMS[1:]},
        matched={k: dict(record=aggregate(lambda v: v["matched"]["policies"][k], keys),
                         direct_vs_anchor=interval(lambda v: v["matched"]["gains_vs_anchor"][k]))
                 for k in ("anchor", "prefix_match", *ARMS)})


def execute(key, roster, credit, rank, previous_matched, source, replay, beat):
    sites = {s: base.load(s, source) for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(v[0]) for s, v in sites.items()}
    tensors = {s: attach_refs(v[3], v[4], v[5], banks[s]) for s, v in sites.items()}
    heads, forecasts, updates = {}, {}, 0
    for held, (_, _, _, context, _, _) in sites.items():
        donors = [s for s in sites if s != held]
        sample = batches([tensors[s] for s in donors])
        for seed in SEEDS:
            cell = f"{held}_{seed}"
            paths = {}
            for arm in ARMS:
                def train_loss(net, ego, peers, mask, scale, y, valid, cv, ols):
                    return loss.objective(arm, net(ego, peers, mask), y, valid, scale, cv, ols)

                name = f"{cell}_{arm}"
                identity = dict(registration=key, held=held, fit=donors, seed=seed, arm=arm)
                net, heads[name], u = base.two.fit.fit(
                    identity, name, seed, lambda: base.model.Forecast("comoving_ego"),
                    STEPS, sample, train_loss, replay, beat)
                updates += u
                paths[arm] = base.predict(net, context)
            forecasts[cell] = freeze_predictions(cell, paths)
            beat(phase="freeze", cell=cell, updates=updates)
    freeze = OUT / "freeze.json"
    run.once(freeze, run.stable(dict(registration=key, heads=heads, predictions=forecasts)))

    result = {}
    for held, (d, rec, starts, _, _, _) in sites.items():
        donors = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in donors]), [.25, .75])
        bank = banks[held]
        packet = base.prior.prepare({s: v[0] for s, v in sites.items()}, held, donors, bank)
        result[held] = []
        for cell in roster["by_site"][held]:
            seed = cell["seed"]
            prediction = run.checked(forecasts[f"{held}_{seed}"])
            with np.load(prediction, allow_pickle=False) as z:
                paths = {k: z[k] for k in ARMS}
            menu, count = base.ledger.frozen_count(
                d, bank, rec, starts, credit["heads"][f"{held}/{seed}/additive_rank"],
                rank["heads"][f"{held}_{seed}_motion"])
            ref = next(v for v in previous_matched["by_site"][held] if v["seed"] == seed)
            result[held].append(dict(seed=seed, rows=len(d["h"]), valid_steps=int(d["valid"].sum()),
                full=full_metrics(d, bank, paths, cuts),
                matched=matched_metrics(d, bank, paths, packet, menu, count, cuts, ref)))
            beat(phase="score", site=held, seed=seed, updates=updates)
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, freeze=run.sha(freeze),
        result_source="fresh_run", by_site=result, summary=summarize(result),
        original_2pct=True, confirmation_access=False, deployment_changed=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true", help="verify only; do not train")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    base.two.fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            state = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", state)
            if v.get("step", STEPS) in (100, STEPS):
                print(state, flush=True)

        u = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=u, replay=args.replay)


if __name__ == "__main__":
    main()
