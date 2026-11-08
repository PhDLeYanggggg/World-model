"""Outer-location-held ablation of learned utility with real or mismatched maps."""

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
from scripts import run_m3w_opendd_map_direction as road
from src.world_model import m3w_map_path_value as model

run, np, torch, source, prior, transfer = road.run, road.np, road.torch, road.source, road.prior, road.transfer
OUT = road.OUT.parent / "opendd_map_path_value_v1"
PRIVATE = road.PRIVATE.parent / OUT.name
STEPS, BATCH = 500, 128
ANCHOR = prior.ARMS.index("tangent")


def register():
    parent, frozen, earlier, maps = road.register()
    p = road.OUT / "analysis.json"
    a = run.read(p)
    if a["registration"] != parent or a["confirmation_access"] or not a["control_replay_exact"]:
        raise ValueError("Require the registered road-context result")
    donor_reg, donors = transfer.parent.old.register()
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        loss=run.artifact(Path(model.cost.__file__)), road_experiment=run.artifact(p), inner_producers=donor_reg,
        split="leave one entire OpenDD location out of all fits and easy thresholds",
        donor_bank="pair-excluded producer; refiner excludes outer location only",
        wrong_map="cyclic exchange among at least two mapped edited agents in the same query",
        matched_anchor="fixed tangent", primary="map vs no_map, wrong_map and fixed tangent at half radius",
        secondary="same-count shuffle stratified by recording and active map support", arms=model.MODES,
        seeds=prior.SEEDS, steps=STEPS, batch=BATCH, threads=4, workers=0, horizon=[8, 12],
        original_2pct=True, confirmation_access=False, no_target_tuning=True)))
    return run.sha(f), frozen, earlier, maps, donors


def load_site(site):
    p = source.prior.PRIVATE / f"{site}_cohort.npz"
    with np.load(p, allow_pickle=False) as z:
        h, y, q, rec = z["history"], z["future"], np.diff(z["offsets"]), z["recording"]
    valid = np.isfinite(y).all(2)
    cv = transfer.motion.rollout(h, 12)[0]
    easy = np.linalg.norm(cv[valid.all(1)]-y[valid.all(1)], axis=-1).mean(1)
    return dict(h=h, y=y, valid=valid, query=np.repeat(np.arange(len(q)), q), rec=np.repeat(rec, q),
                cut_samples=easy[easy > 0])


def prepare(d, forecaster, raster, fraction):
    cv, ols, anchor, _, delta = forecaster
    p, on = transfer.matched_paths(cv, anchor, delta, fraction)
    bank = np.stack([p[k] for k in prior.ARMS], axis=1)
    x, radius = model.cost.features(d["h"][on], cv[on], bank[on])
    z, support = model.map_features(d["h"][on], cv[on], bank[on], radius, raster)
    real, wrong, exchanged = model.wrong_map(z, d["query"][on], support)
    return dict(cv=cv, ols=ols, paths=p, on=on, bank=bank, radius=radius, x=x, real=real, wrong=wrong,
                mapped=int(support.sum()), exchanged=exchanged)


def design(x, real, wrong, arm):
    m = torch.zeros_like(real) if arm == "no_map" else real if arm == "map" else wrong
    return torch.cat((x, m), dim=-1)


def fit(key, held, seed, arm, donors, fit_sites, beat, replay):
    ident = dict(registration=key, held=held, seed=seed, arm=arm, donors=fit_sites)
    file = PRIVATE / f"{held}_{seed}_{arm}.pt"
    torch.manual_seed(seed)
    head = model.PathValue()
    opt = torch.optim.AdamW(head.parameters(), lr=.001)
    rng = np.random.default_rng(seed+8209)
    step, trace = 0, []
    if file.exists():
        state = torch.load(file, map_location="cpu", weights_only=True)
        if state["identity"] != ident or state["step"] not in range(100, STEPS+1, 100):
            raise ValueError("Held location or checkpoint identity changed")
        step, trace = state["step"], state["trace"]
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot train")
    for t in range(step+1, STEPS+1):
        which = rng.integers(len(donors), size=BATCH)
        pick = [rng.choice(len(d[0]), size=int((which == j).sum())) for j, d in enumerate(donors)]
        x, real, wrong, y = (torch.cat([donors[j][k][a] for j, a in enumerate(pick)]) for k in range(4))
        loss = model.cost.loss(head(design(x, real, wrong, arm)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = file.with_suffix(".pending")
            torch.save(dict(identity=ident, head=head.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, step=t, trace=trace), tmp)
            tmp.replace(file)
            beat(phase="fit", site=held, seed=seed, arm=arm, step=t)
    return head.eval(), dict(identity=ident, artifact=run.artifact(file), loss=trace), STEPS-step


@torch.inference_mode()
def evaluate(d, p, nets, cuts, historical):
    cv, bank, on = p["cv"], p["bank"], p["on"]
    values = dict(cv=cv, ols4=p["ols"], hindcast=p["paths"]["hindcast"], tangent=p["paths"]["tangent"])
    x, a, b = (torch.from_numpy(p[k]) for k in ("x", "real", "wrong"))
    choice = {}
    for k, net in nets.items():
        pred = net(design(x, a, b, k)).numpy()
        c = np.full(len(cv), ANCHOR)
        c[on] = model.cost.select(pred, ANCHOR)["agreement"]
        choice[k] = c
        values[k] = bank[np.arange(len(bank)), c]
        np.testing.assert_array_equal((values[k] != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(values[k][on]-cv[on], axis=-1).max(1),
                                   p["radius"][:, 0], atol=1e-13, rtol=1e-7)
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    a = prior.score(values, bank, on, {}, choice, y, valid, rec, cuts)
    for k in prior.CONTROL:
        if a["arms"][k] != historical["arms"][k]:
            raise ValueError("Matched baseline replay changed")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    obs = {f"map_vs_{k}": float(100*transfer.paired(values[k], values["map"], y, valid)[1].sum()/mass)
           for k in ("no_map", "wrong_map")}
    support = np.zeros(len(cv), dtype=bool)
    support[on] = p["exchanged"]
    strata = np.char.add(np.char.add(rec, ":"), support.astype(int).astype(str))
    sh = {k: road.shuffle.cell(cv, bank, on, c, y, valid, strata) for k, c in choice.items()}
    return dict(arms=a["arms"], direct=obs, shuffle=sh, edits=a["edits"], mapped=p["mapped"],
                aligned_or_exchanged=int(support.sum()), path_counts=a["path_choices"])


def summarize(cells, part):
    draw = np.random.default_rng(817).integers(len(cells), size=(3000, len(cells)))

    def stat(fn):
        a = np.array([np.mean([fn(x[part]) for x in group]) for group in cells.values()])
        return dict(point=float(a.mean()), interval95=np.quantile(a[draw].mean(1), [.025, .975]).tolist())

    rows = [v[part] for g in cells.values() for v in g]
    return dict(arms={k: dict(vs_cv=stat(lambda r: r["arms"][k]["lower"]["cv"]),
        vs_tangent=stat(lambda r: r["arms"][k]["lower"]["tangent"]),
        passes=sum(r["arms"][k]["pass_views"] for r in rows), undefined=sum(r["arms"][k]["undefined_easy"] for r in rows),
        views=sum(r["arms"][k]["views"] for r in rows)) for k in (*prior.CONTROL, *model.MODES)},
        direct={k: stat(lambda r: r["direct"][k]) for k in rows[0]["direct"]},
        shuffle={k: stat(lambda r: r["shuffle"][k]["direct_lower"]) for k in model.MODES},
        support={k: sum(r[k] for r in rows) for k in ("edits", "mapped", "aligned_or_exchanged")})


def execute(key, frozen, earlier, maps, donors, beat, replay, scratch):
    frozen_models = transfer.frozen_models(*frozen[:5])
    _, inner = transfer.parent.old.load_producers(torch, transfer.parent.producer, donors)
    data = {s: load_site(s) for s in source.SITES}
    raster = {s: road.raster_for(s, maps[s])[0] for s in source.SITES}
    out, heads, updates = {}, {}, 0
    for j, held in enumerate(source.SITES):
        fit_sites = [s for s in source.SITES if s != held]
        cuts = np.quantile(np.concatenate([data[s]["cut_samples"] for s in fit_sites]), [.25, .75])
        rows = []
        for i, seed in enumerate(prior.SEEDS):
            p = scratch / f"{held}_{seed}.json"
            if not p.exists():
                outer, refiners = frozen_models[seed][j]
                pool = []
                for s in fit_sites:
                    d = data[s]
                    donor = inner[tuple(sorted((held, s))), seed]
                    v = prepare(d, transfer.forecast(d["h"], [(donor, refiners)]), raster[s], .5)
                    on = v["on"]
                    y = model.cost.targets(v["cv"][on], v["bank"][on], v["radius"], d["y"][on], d["valid"][on], cuts[0])
                    pool.append(tuple(torch.from_numpy(a) for a in (v["x"], v["real"], v["wrong"], y)))
                nets, meta = {}, {}
                for arm in model.MODES:
                    nets[arm], meta[arm], n = fit(key, held, seed, arm, pool, fit_sites, beat, replay)
                    updates += n
                d = data[held]
                pred = transfer.forecast(d["h"], [(outer, refiners)])
                hist = earlier["groups"]["opendd"][held][i]
                res = {k: evaluate(d, prepare(d, pred, raster[held], f), nets, cuts, hist[k])
                       for k, f in transfer.FRACTIONS.items()}
                run.once(p, run.stable(dict(registration=key, held=held, seed=seed, heads=meta, **res)))
            v = run.read(p)
            if v["registration"] != key or v["held"] != held or v["seed"] != seed:
                raise ValueError("Checkpoint outer role mismatch")
            for arm, receipt in v["heads"].items():
                run.checked(receipt["artifact"])
                heads[f"{held}/{seed}/{arm}"] = receipt
            rows.append({k: a for k, a in v.items() if k != "heads"})
            beat(phase="held_score", site=held, seed=seed)
        out[held] = rows
    if len(heads) != 63:
        raise ValueError("Require all three ablations, locations and seeds")
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=heads, held_labels_used_in_fit=False)))
    report = {k: summarize(out, k) for k in transfer.FRACTIONS}
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(f), summary=report, by_site=out,
        original_2pct=True, held_location_fully_excluded=True, donor_refiner_can_include_donor=True,
        no_target_tuning=True, confirmation_access=False, deployment=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v.get("phase") != "fit":
                print(row, flush=True)

        state = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as t:
                updates = execute(*state, beat, True, Path(t))
        else:
            updates = execute(*state, beat, False, PRIVATE)
        beat(phase="complete", updates=updates, replay=args.replay)


if __name__ == "__main__":
    main()
