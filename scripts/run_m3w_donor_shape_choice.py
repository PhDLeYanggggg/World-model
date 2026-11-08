"""Fit and cross-validate risk-aware shape choices without changing recipients."""

import argparse
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_donor_motion_shape as prior
from scripts import run_m3w_two_edit as two
from src.world_model import m3w_donor_shape_choice as model

run, parent, paired, ledger = prior.run, prior.parent, prior.paired, prior.ledger
OUT = prior.OUT.parent / "donor_shape_choice_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS = 300
ARMS = ("anchor", "prefix_match", "gain_only", "risk_aware", "utility_oracle")
RISK = ("all_excess", "easy_excess", "easy_signed")


def register():
    key, a, ph, h, _, source = prior.register()
    p = prior.OUT / "analysis.json"
    old = run.read(p)
    if old["registration"] != key or old["confirmation_access"] or not old["original_2pct"]:
        raise ValueError("Require the registered matched shape experiment")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parent=run.artifact(p), native_steps=[8, 12],
        steps=STEPS, workers=0, threads=4, seeds=two.study.prior.SEEDS, batch=128,
        arms=ARMS, margins=model.MARGINS, penalties=model.PENALTIES,
        primary="risk-aware minus prefix-match, direct lower utility and original full-roster joint passes",
        target="lower paired gain vs original trajectory and three changes in original risk numerators",
        fitting="site-balanced; each train row excludes its own site and every model-held site from retrieval and cuts",
        calibration="double-excluded predictions on legal donor row/amplitude options, not frozen controller nominees; "
                    "maximize donor macro lower gain with nonpositive macro changes in each risk numerator for guarded arm",
        budget="fixed outer rows, query times and radii for every arm; oracle maximizes direct lower utility only",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        original_2pct=True, confirmation_access=False)))
    return run.sha(f), a, ph, h, old, source


def prepare(sites, site, use, bank):
    d = sites[site]
    cv, _, _, _, scale, legal = bank
    history = d["h"]
    donors = {s: sites[s] for s in use}
    pred = {k: prior.model.Library.fit(donors, site, k).predict(history[:, :k])
            for k in (*prior.model.PREFIXES, 8)}
    paths, chosen = prior.choices(history, legal, {k: v[0] for k, v in pred.items()})
    row, arm = np.nonzero(legal)
    amp = np.array([.25, .5, 1.])[arm]
    anchor, donor = paths[row, arm, :2].swapaxes(0, 1)
    x = model.features(history[row], cv[row], anchor, donor, amp, pred[8][1][row], chosen[row, arm, 0])
    index = np.full(legal.shape, -1)
    index[row, arm] = np.arange(len(row))
    return dict(paths=paths, row=row, arm=arm, index=index, x=torch.from_numpy(x), scale=scale[row])


def targets(d, bank, packet, cut):
    row, arm = packet["row"], packet["arm"]
    cv = bank[0][row]
    y, valid = d["y"][row], d["valid"][row]
    anchor, donor = packet["paths"][row, arm, :2].swapaxes(0, 1)
    gain = paired(anchor, donor, y, valid)[1].sum(1)
    a, b = [two.moments(cv, p, y, valid, cut) for p in (anchor, donor)]
    delta = np.column_stack([gain, *[12.*(b[k]-a[k]) for k in RISK]])
    mass = float(np.linalg.norm(cv-y, axis=-1)[valid].sum())
    if not np.isfinite(delta).all() or mass <= 0.:
        raise ValueError("Invalid donor target or denominator")
    return delta, mass


@torch.inference_mode()
def evaluate(d, bank, packet, menu, count, net, cuts, rules, ref):
    cv, ols = bank[:2]
    use = np.arange(2) < count[:, None]
    at, arm = menu["row"][use], menu["arm"][use]
    index = packet["index"][at, arm]
    if (index < 0).any():
        raise ValueError("Frozen intervention is not legal")
    path = packet["paths"][at, arm]
    pred = net(packet["x"][index]).numpy()
    y, valid = d["y"], d["valid"]
    p = {k: cv.copy() for k in ARMS}
    p["anchor"][at], p["prefix_match"][at] = path[:, 0], path[:, 3]
    take = {k: model.use_donor(pred, rule) for k, rule in rules.items()}
    direct = paired(path[:, 0], path[:, 1], y[at], valid[at])[1].sum(1)
    take["utility_oracle"] = direct > 0.
    for k, v in take.items():
        p[k][at] = path[np.arange(len(path)), v.astype(int)]
    for forecast in p.values():
        np.testing.assert_array_equal(np.flatnonzero((forecast != cv).any((1, 2))), np.sort(at))
        np.testing.assert_allclose(prior.model.peak(forecast[at]-cv[at]),
                                  prior.model.peak(p["anchor"][at]-cv[at]), atol=1e-10)
    result = {k: parent.score(d, cv, ols, v, cuts) for k, v in p.items()}
    for k in ("anchor", "prefix_match"):
        for field, value in ref["policies"][k].items():
            np.testing.assert_allclose(result[k][field], value, atol=1e-10, rtol=0.)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {f"{k}_vs_{b}": float(100.*paired(p[b], p[k], y, valid)[1].sum()/mass)
               for k in ARMS[2:] for b in ARMS[:2]}
    calibration = dict(pred_gain_mae=float(np.abs(pred[:, 0]-direct/(12.*packet["scale"][index])).mean())
                       if len(at) else None)
    return dict(policies=result, effects=effects, chosen={k: int(v.sum()) for k, v in take.items()},
                supported_gain_count=int((direct > 0.).sum()), prediction_diagnostic=calibration)


def execute(key, a, ph, h, old, source, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    summaries, settings, heads, updates = {}, {}, {}, 0

    @lru_cache(maxsize=16)
    def packet(site, use):
        return prepare(sites, site, use, banks[site])

    @lru_cache(maxsize=72)
    def cuts(use):
        return np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])

    @lru_cache(maxsize=24)
    def dataset(site, use):
        p = packet(site, use)
        delta, mass = targets(sites[site], banks[site], p, cuts(use)[0])
        target = (delta/(12.*p["scale"][:, None])).clip(-8., 8.).astype(np.float32)
        return p["x"], torch.from_numpy(target), delta, mass

    @lru_cache(maxsize=8)
    def fitted(excluded, seed):
        nonlocal updates
        use = tuple(s for s in sites if s not in excluded)
        data = [dataset(s, tuple(u for u in use if u != s))[:2] for s in use]
        name = "_".join(excluded)+f"_{seed}"
        identity = dict(registration=key, excluded=list(excluded), fit=list(use), seed=seed)
        net, meta, n = two.fit.fit(identity, name, seed, lambda: model.network(data[0][0].shape[1]),
                                   STEPS, two.batcher(data), model.loss, replay, beat)
        heads[name] = meta
        updates += n
        return net

    for held, d in sites.items():
        donors = tuple(s for s in sites if s != held)
        seeds = [x["seed"] for x in a["by_site"][held]]
        cells = {}
        for s in donors:
            use = tuple(u for u in donors if u != s)
            x, _, delta, mass = dataset(s, use)
            cells[s] = []
            for seed in seeds:
                net = fitted(tuple(sorted((held, s))), seed)
                with torch.inference_mode():
                    cells[s].append((net(x).numpy(), delta, mass))
        gain, table_gain = model.calibrate(cells, guarded=False)
        risk, table_risk = model.calibrate(cells, guarded=True)
        rules = dict(gain_only=gain, risk_aware=risk)
        settings[held] = dict(rules=rules, gain_grid=table_gain, risk_grid=table_risk)
        p = packet(held, donors)
        with np.load(source.PRIVATE / f"{held}_cohort.npz", allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        summaries[held] = []
        for seed in seeds:
            net = fitted((held,), seed)
            menu, count = ledger.frozen_count(d, banks[held], rec, starts,
                ph["heads"][f"{held}/{seed}/additive_rank"], h["heads"][f"{held}_{seed}_motion"])
            ref = next(x for x in old["by_site"][held]["cells"] if x["seed"] == seed)
            result = evaluate(d, banks[held], p, menu, count, net, cuts(donors), rules, ref)
            summaries[held].append(dict(seed=seed, **result))
        beat(phase="held", site=held, rules=rules, updates=updates)

    flat = [r for v in summaries.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    effects = {}
    for k in flat[0]["effects"]:
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in summaries.values()])
        effects[k] = dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                          worst_site=float(x.min()))
    totals = {k: {field: sum(c["policies"][k][field] for c in flat)
                 for field in ("edits", "views", "undefined", "risk_pass", "dual_positive")} for k in ARMS}
    chosen = {k: sum(c["chosen"][k] for c in flat) for k in ARMS[2:]}
    freeze = OUT / "head_freeze.json"
    run.once(freeze, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(freeze),
        by_site=summaries, settings=settings, summary=dict(effects=effects, policies=totals, chosen=chosen),
        original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    two.fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            if v.get("step", STEPS) == STEPS:
                print(z, flush=True)

        updates = execute(*register(), args.replay, beat)
        beat(complete=True, updates=updates)


if __name__ == "__main__":
    main()
