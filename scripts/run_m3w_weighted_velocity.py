"""Train source-separated causal weighted velocity and compare full-strength paths."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_optional_allocation as parent
from src.world_model import m3w_weighted_velocity as model

direction, job, run, np, torch = parent.direction, parent.job, parent.run, parent.np, parent.torch
OUT = parent.OUT.parent / "weighted_velocity_v1"
PRIVATE = parent.PRIVATE.parent / OUT.name
HEADS = ("ade", "dual")
STEPS = 600
BASE = ("quarter", "ols4", "ols8", "fixed_robust", *HEADS)
WHOLE = tuple(f"whole_{k}" for k in BASE if k != "quarter")
ARMS = (*BASE, *WHOLE, "cal_static", "cal_learned")
PAIRS = {f"{a}_vs_{'cv' if b == job.CV else b}": (b, a, job.CV)
         for a, b in ((job.OLS, job.CV), *((a, b) for a in ARMS for b in (job.CV, job.OLS)),
                      ("dual", "ade"), ("dual", "fixed_robust"), ("dual", "ols4"), ("dual", "quarter"),
                      ("whole_dual", "whole_ade"), ("whole_dual", "whole_fixed_robust"))}


def register():
    parent_key, identity, cut, freeze = parent.register()
    p = parent.OUT / "ols_limit.json"
    o = run.read(p)
    if o["registration"] != parent_key or o["confirmation_access"]:
        raise ValueError("Require the registered OLS4 ceiling")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=parent_key, sparse_ceiling=run.artifact(p), seeds=job.SEEDS, steps=STEPS,
        checkpoint_every=100, cpu_threads=4, workers=0, batch=128, lr=.001,
        source_roles="six fit, two inner calibration, one outer development source",
        matched_recipients="earlier frozen query gain ranking", forecast_origin="last observed point",
        heads=list(HEADS), primary="dual_vs_ade_direct_paired_completion_lower",
        whole_roster_arms_are_diagnostic=True, original_2pct=True, confirmation_access=False,
        official_calibration_access=False)))
    return run.sha(f), identity, cut, freeze


def tensors(identity, beat):
    x, truth, blocks = job.prior.prior.prepare(identity, beat)
    s = job.prior.prior.prior.base.train_source.fitting_only(identity, beat)
    h = s["geometry"][:, :16].reshape(-1, 8, 2).copy()
    for b in blocks:
        sl = b["sl"]
        np.testing.assert_array_equal(s["geometry"][sl, 332:356], b["cv"].reshape(-1, 24))
        np.testing.assert_array_equal(s["target"][sl], b["target"])
        np.testing.assert_array_equal(s["valid"][sl], b["valid"])
    stack = lambda k: torch.from_numpy(np.concatenate([b[k] for b in blocks]).astype("float32"))
    fit_data = (torch.from_numpy(h), stack("cv"), stack("ols"), stack("target"),
                torch.from_numpy(s["valid"].copy()), stack("scale"))
    return x[0], truth[-1], blocks, fit_data


def fit(key, fold, name, data, raw, groups, beat, replay):
    fit_ids = [groups[g] for g in fold["fit"]]
    v = raw[np.concatenate(fit_ids)]
    cut = float(np.quantile(v[np.isfinite(v) & (v > 0.)], .25))
    easy = torch.from_numpy(np.isfinite(raw) & (raw > 0.) & (raw <= cut))
    held, seed = fold["held"], fold["seed"]
    identity = dict(key=key, held=held, seed=seed, name=name, train=fold["fit"],
                    calibration=fold["calibration"], easy_cut=cut)
    p = PRIVATE / f"{name}_{held}_{seed}.pt"
    torch.manual_seed(seed)
    net = model.WeightedVelocity()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, trace = 0, []
    if p.exists():
        st = torch.load(p, weights_only=True, map_location="cpu")
        if st["identity"] != identity or not 0 < st["step"] <= STEPS:
            raise ValueError("Unexpected weight-fit identity")
        net.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        rng.set_state(st["rng"])
        step, trace = st["step"], st["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot train")
    for step in range(step+1, STEPS+1):
        i = direction.timing.draw_batch(fit_ids, 128, rng)
        h, cv, ols, truth, valid, scale = (z[i] for z in data)
        value = model.objective(net(h)[0], cv, ols, truth, valid, scale, easy[i], name == "dual")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            trace.append(float(value.detach()))
            temp = p.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, trace=trace, model=net.state_dict(),
                optimizer=optimizer.state_dict(), rng=rng.get_state()), temp)
            temp.replace(p)
            beat(stage="checkpoint", held=held, seed=seed, name=name, step=step, loss=trace[-1])
    return net.eval(), run.artifact(p)


@torch.inference_mode()
def forecasts(cell, h, heads):
    b, i, _, d, _, _ = cell
    history = h[b["sl"]]
    full = {"ols4": b["ols"], "ols8": model.anchored_path(model.line_fit(
        history.to(torch.float64), torch.ones_like(history[..., 0], dtype=torch.float64))[0]).numpy(),
        "fixed_robust": model.robust_path(history.to(torch.float64)).numpy()}
    weights = {}
    for k, net in heads.items():
        p, w = net(history)
        full[k] = p.numpy().astype(float)
        weights[k] = (w/w.sum(1, keepdim=True)).numpy()
    paths = {f"whole_{k}": p for k, p in full.items()}
    for k, p in full.items():
        v = b["cv"].copy()
        v[i] = p[i]
        paths[k] = v
    paths["quarter"] = b["cv"].copy()
    paths["quarter"][i] += .25*d
    return paths, weights


def calibration(cells, cut):
    trials = {}
    for k in BASE:
        bb, choices = [], []
        for b, paths in cells:
            p = paths[k]
            bb.append(b)
            choices.append((np.any(p != b["cv"], axis=(1, 2)), job.calibration_cell(b, p, cut)))
        trials[k] = job.trial(bb, [np.zeros(len(b["cv"])) for b in bb], choices, 0.)
    choose = direction.timing.select
    return dict(static=choose(trials, BASE[:-2]), learned=choose(trials, HEADS)), trials


def execute(key, identity, cut, freeze, beat, replay):
    x, raw, blocks, data = tensors(identity, beat)
    groups = job.prior.prior.group_ids(blocks)
    rows, fits, weight_rows = [], [], []
    for fold in freeze["calibration"]:
        held, seed = fold["held"], fold["seed"]
        if job.model.roles(groups, held) != (fold["fit"], fold["calibration"]):
            raise ValueError("Source roles overlap")
        sets, nominee = direction.timing.inputs(x, blocks, fold, freeze["registration"])
        networks = {k: fit(key, fold, k, data, raw, groups, beat, replay) for k in HEADS}
        heads = {k: v[0] for k, v in networks.items()}
        cal = [(cell[0], forecasts(cell, data[0], heads)[0])
               for cell in sets if cell[0]["source"] in fold["calibration"]]
        chosen, trials = calibration(cal, cut)
        receipt = run.stable(dict(key=key, held=held, seed=seed, fit=fold["fit"], calibration=fold["calibration"],
            nominee=nominee, checkpoints={k: v[1] for k, v in networks.items()}, choices=chosen, trials=trials))
        f = PRIVATE / f"policy_{held}_{seed}.json"
        if replay and not f.exists():
            raise ValueError("Replay may not select on calibration")
        run.once(f, receipt)
        fits.append(receipt)
        for cell in sets:
            b, i = cell[:2]
            if b["source"] != held:
                continue
            pp, weights = forecasts(cell, data[0], heads)
            pp.update({job.CV: b["cv"], job.OLS: b["ols"]})
            for k, arm in chosen.items():
                pp[f"cal_{k}"] = pp[arm or job.CV]
            row = direction.SCORE(pp, PAIRS, b["target"], b["valid"], b["inverse"], b["rec"],
                easy_cut=cut["easy"], hard_cut=cut["hard"])
            rows.append(dict(row, source=held, seed=seed))
            weight_rows.append(dict(recording=b["rec"], source=held, seed=seed, nominated=len(i),
                mean_weights={k: v.mean(0).tolist() for k, v in weights.items()},
                unchanged_matched_paths={k: len(i)-row["metrics"][k]["risk"]["selected"] for k in BASE}))
    if len(fits) != 27 or len(rows) != 72:
        raise ValueError("Require 54 trained heads and 72 held views")
    summary = job.grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, job.SEEDS)
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, fits=fits, weight_rows=weight_rows,
        summary=summary, official_calibration_access=False, confirmation_access=False, deployment=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, identity, cut, freeze = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            if v.get("step") == STEPS:
                print(v, flush=True)

        execute(key, identity, cut, freeze, beat, args.replay)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
