"""Compare finite and CV-tangent losses at a matched OpenDD intervention radius."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_radius_shape as old
from scripts import run_m3w_opendd_weighted_velocity as anchor
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_cv_tangent as model
from src.world_model import m3w_opendd_velocity_candidate as producer, m3w_radius_shape as shape

job, run, torch, np = anchor.job, anchor.run, anchor.torch, anchor.np
OUT = anchor.OUT.parent / "opendd_cv_tangent_v1"
PRIVATE = anchor.PRIVATE.parent / OUT.name
STEPS = 600
HEADS = ("finite", "tangent")
RADIUS = .5
GATE = .6
PAIRS = (("finite", "half"), ("tangent", "half"), ("tangent", "finite"),
         ("tangent", "previous_shape"), ("guard", "matched_half"), ("guard", "matched_previous"))


def register():
    _, caches = anchor.register()
    parent, donors = old.register()
    old_results = old.PUBLIC / "analysis.json"
    freeze = old.PUBLIC / "checkpoint_freeze.json"
    if run.read(old_results)["registration"] != parent or run.read(freeze)["registration"] != parent:
        raise ValueError("Require the frozen stronger OpenDD controls")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        producer_registration=parent, prior_shape_results=run.artifact(old_results),
        prior_shape_checkpoints=run.artifact(freeze), weighted_velocity=run.artifact(anchor.OUT / "analysis.json"),
        caches=caches, seeds=job.SEEDS, steps=STEPS, threads=4, workers=0, radius_multiplier=RADIUS,
        optional_guard_probability=GATE, guard_is_not_calibrated=True,
        full_grid_open_dd_primary=True, original_recording_2pct=True,
        confirmation_access=False, formal_calibration_access=False)))
    return run.sha(f), donors, run.read(freeze)["fits"]


def fit(key, site, seed, arm, sources, c, beat, replay):
    donors = sorted(s for s in sources if s != site)
    cuts = old.base.cuts_for([sources[s] for s in donors])
    bank, pool, n = [], [], 0
    for s in donors:
        d, p = sources[s], c[s]
        x, unit = p["x"], p["unit"]
        cv = torch.from_numpy(((d["cv"]@d["rot"])/d["scale"][:, None, None]).astype("float32"))
        truth = torch.from_numpy(((d["y"]@d["rot"])/d["scale"][:, None, None]).astype("float32"))
        radius = torch.from_numpy((RADIUS*p["ratio"].numpy()).astype("float32"))
        valid = torch.from_numpy(d["valid"])
        easy = torch.from_numpy((d["cv_ade"] > 0.) & (d["cv_ade"] <= cuts[0]))
        bank.append((x, unit, radius, cv, truth, valid, easy))
        pool.append(p["pool"]+n)
        n += len(x)
    data = list(map(torch.cat, zip(*bank, strict=True)))
    identity = dict(key=key, held=site, seed=seed, arm=arm, fit=donors)
    f = PRIVATE / f"{site}_{seed}_{arm}.pt"
    torch.manual_seed(seed)
    net = model.CVTangent(data[0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, trace = 0, []
    if f.exists():
        state = torch.load(f, map_location="cpu", weights_only=True)
        if state["identity"] != identity or not 0 < state["step"] <= STEPS:
            raise ValueError("Checkpoint roles differ from registered fit")
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        step, trace = state["step"], state["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot fit")
    for step in range(step+1, STEPS+1):
        i = anchor.study.direction.timing.draw_batch(pool, 128, rng)
        x, unit, radius, cv, y, valid, easy = (v[i] for v in data)
        step_path, logit = net(x, unit, radius)
        loss = model.losses(step_path, logit, cv, y, valid, easy, arm == "tangent")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = f.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, trace=trace, model=net.state_dict(),
                optimizer=opt.state_dict(), rng=rng.get_state()), tmp)
            tmp.replace(f)
            beat(held=site, seed=seed, arm=arm, step=step, loss=trace[-1])
    return net.eval(), cuts, run.artifact(f)


@torch.inference_mode()
def paths(d, c, fitted, site, seed, parent):
    cv = d["cv"]
    r = RADIUS*c["r"]
    old_meta = parent[f"{site}/{seed}/easy_guard"]
    checkpoint = old.PRIVATE / f"{site}_{seed}_easy_guard.pt"
    if run.sha(checkpoint) != old_meta["sha256"]:
        raise ValueError("Previous shape control changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["identity"] != old_meta["identity"] or state["identity"]["held"] != site:
        raise ValueError("Previous shape control must exclude this location")
    prior = producer.network(c["x"].shape[1])
    prior.load_state_dict(state["model"])
    on = r > 0.
    previous = np.zeros_like(cv)
    previous[on] = shape.shape(prior.eval()(c["x"][on]), c["unit"][on]).numpy()
    world = lambda v: v@d["rot"].transpose(0, 2, 1)
    out = {"cv": cv, "half": cv+RADIUS*(c["inc"]-cv),
           "previous_shape": cv+shape.at_radius(world(previous), r)}
    for arm, (net, _, _) in fitted.items():
        pred, logit = net(c["x"], c["unit"], torch.ones(len(r)))
        out[arm] = cv+shape.at_radius(world(pred.numpy().astype(float)), r)
        if arm == "tangent":
            guard = (logit.sigmoid().numpy() >= GATE) & (r > 0)
    for k in ("half", "previous_shape", *HEADS):
        np.testing.assert_allclose(np.linalg.norm(out[k]-cv, axis=-1).max(1), r, rtol=1e-7, atol=1e-14)
    for p, k in (("guard", "tangent"), ("matched_half", "half"), ("matched_previous", "previous_shape")):
        out[p] = np.where(guard[:, None, None], out[k], cv)
        np.testing.assert_array_equal((out[p] != cv).any((1, 2)), guard)
    return out, r


def score(d, values, r, cuts):
    cv = d["cv"]
    out = {}
    for k, p in values.items():
        v = policy(p, cv, cv, d["y"], d["valid"], d["recordings"], easy_cut=cuts[0], hard_cut=cuts[1])
        risk = list(v["recordings"].values())
        out[k] = dict(vs_cv=old.base.paired_summary(d, p, cv, r > 0), edits=int((p != cv).any((1, 2)).sum()),
            passes=sum(r["supported"] for r in risk), views=len(risk),
            undefined_easy=sum(r["completion_easy_risk_upper"] is None for r in risk),
            all_risk_violations=sum(r["completion_all_risk_upper"] is not None and
                                    r["completion_all_risk_upper"] > .02 for r in risk),
            worst_easy_percent=max((100*r["known_easy_net_degradation"] for r in risk
                                    if r["known_easy_net_degradation"] is not None), default=None),
            hard_observed_gain_percent=v["subsets"]["hard"]["gain_percent"],
            ade_available=v["known_ADE_pixel"], fde_available=v["available_endpoint_FDE_diagnostic_pixel"],
            p95_available_ade=v["p95_known_ADE_pixel"])
    pairs = {f"{a}_vs_{b}": old.base.paired_summary(d, values[a], values[b], r > 0) for a, b in PAIRS}
    return dict(policies=out, pairs=pairs, relative=old.pair_errors(d, values))


def summarize(rows):
    draws = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))
    order = list(rows)
    def ci(fn):
        x = np.array([np.mean([fn(v) for v in rows[s]]) for s in order])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    flat = [r for v in rows.values() for r in v]
    summary = {}
    for k in flat[0]["policies"]:
        summary[k] = {f: sum(v["policies"][k][f] for v in flat) for f in
                      ("edits", "passes", "views", "undefined_easy", "all_risk_violations")}
        summary[k]["vs_cv"] = ci(lambda r: r["policies"][k]["vs_cv"]["lower_pp"])
        summary[k]["worst_easy_percent"] = max((v["policies"][k]["worst_easy_percent"] for v in flat
            if v["policies"][k]["worst_easy_percent"] is not None), default=None)
    return dict(policies=summary, direct={k: ci(lambda r: r["pairs"][k]["lower_pp"]) for k in flat[0]["pairs"]},
        tangent_minus_finite_relative_error=ci(lambda r: r["relative"]["mean_relative_vector_error"]["tangent"]
                                               -r["relative"]["mean_relative_vector_error"]["finite"]))


def execute(key, donors, previous, beat, replay):
    outer, inner = old.load_producers(torch, producer, donors)
    sources = {s: old.site_data(s, torch, producer, shape) for s in old.SITES}
    rows, fits = {}, []
    for pos, site in enumerate(old.SITES):
        rows[site] = []
        d = sources[site]
        for seed in job.SEEDS:
            c = {s: old.candidate(sources[s], inner[tuple(sorted((s, site))), seed], torch, producer)
                 for s in old.SITES if s != site}
            fitted = {k: fit(key, site, seed, k, sources, c, beat, replay) for k in HEADS}
            fits.extend(v[2] for v in fitted.values())
            vals, r = paths(d, old.candidate(d, outer[seed][pos], torch, producer), fitted, site, seed, previous)
            rows[site].append(dict(seed=seed, **score(d, vals, r, fitted["finite"][1])))
    if len(fits) != 42:
        raise ValueError("Incomplete matched experiment")
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, models=fits, summary=summarize(rows),
        by_location=rows, original_2pct_unchanged=True, guard_selected_on_held=False,
        confirmation_access=False, deployment=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    args = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, donors, previous = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 2))
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            if v["step"] == STEPS:
                print(v, flush=True)

        execute(key, donors, previous, beat, args.replay)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
