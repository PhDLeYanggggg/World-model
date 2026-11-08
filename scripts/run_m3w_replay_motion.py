"""Fit and evaluate a registered, held-source causal forecast mixture."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use the native arm64 .venv-pytorch interpreter.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_ols_anchor_denoising as parent
from src.world_model import m3w_replay_motion as model
from src.world_model.m3w_local_velocity import scores

base, run, prev, np, torch = parent.base, parent.run, parent.prev, parent.np, parent.torch
PUBLIC = parent.PUBLIC.parent / "native_replay_motion_v1"
PRIVATE = parent.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
FILES = (Path(__file__), Path(model.__file__), ROOT / "tests/test_m3w_replay_motion.py",
         PUBLIC / "protocol.md")
CFG = dict(batch=128, steps=800, checkpoint=100, lr=.001, torch_threads=4, workers=0)
PAIRS = tuple((a, "ols4") for a in (*model.ARMS, "cv", "ols8", "median", "bounded_trend")) + (
    ("replay", "cv"), ("replay", "history"), ("replay", "global_mix"))


def register():
    cut, jobs = parent.register()
    links = (*FILES, parent.REG, Path(parent.__file__), Path(scores.__code__.co_filename),
             base.fit_source.PUBLIC / "fit_population.json",
             PUBLIC.parent / "native_velocity_objectives_v1/replay_execution.json")
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in links], cfg=CFG,
        seeds=prev.SEEDS, arms=model.ARMS, experts=model.EXPERTS, comparisons=PAIRS,
        bootstrap=3000, held_source=True, development_targets=False, external_access=False,
        original_rule_unchanged=True)))
    return cut, jobs


def fit(features, deltas, floor, target, valid, scale, train, groups, ident, beat, resume):
    seed = ident["seed"]
    torch.manual_seed(seed)
    net = model.ReplayMotion()
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed + 7919)
    path = PRIVATE / f'{ident["held"]}_{seed}_{ident["arm"]}.pt'
    step, draws, trace = 0, np.zeros(len(train), np.int64), []
    if path.exists():
        if not resume:
            raise FileExistsError("Existing fold requires --resume")
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["identity"] != ident or not 0 < st["step"] <= CFG["steps"]:
            raise ValueError("Checkpoint identity changed")
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        rng.set_state(st["rng"])
        step, draws, trace = (st[k] for k in ("step", "draws", "trace"))
    while step < CFG["steps"]:
        i = base.native.draw_batch(groups, CFG["batch"], rng)
        at = train[i]
        prediction = model.mix(net(features[at]), deltas[at])
        cost = model.objective(prediction, floor[at], target[at], valid[at], scale[at])
        if not torch.isfinite(cost):
            raise FloatingPointError("Non-finite forecast loss")
        opt.zero_grad(set_to_none=True)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        step += 1
        np.add.at(draws, i, 1)
        if step % CFG["checkpoint"] == 0:
            trace.append(dict(step=step, loss=float(cost.detach())))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, model=net.state_dict(),
                optimizer=opt.state_dict(), rng=rng.get_state(), draws=draws, trace=trace), tmp)
            tmp.replace(path)
            beat(state="checkpoint", seed=seed, arm=ident["arm"], held=ident["held"], **trace[-1])
    if draws.sum() != CFG["steps"] * CFG["batch"]:
        raise ValueError("Training draws changed")
    return net.eval(), run.artifact(path), (draws, rng.get_state()), trace


def train_folds(jobs, beat, resume):
    d = parent.data(jobs, beat)
    ego, replay, delta, scale, paths = model.design(d["geometry"][:, :16].reshape(-1, 8, 2))
    np.testing.assert_allclose(paths["cv"], d["geometry"][:, 332:356].reshape(-1, 12, 2), atol=.0001)
    delta, floor, target, valid, scale = map(torch.from_numpy, (delta, paths["ols4"].astype(np.float32),
        d["target"], d["valid"], scale))
    x = {a: torch.from_numpy(model.arm_features(ego, replay, a)) for a in model.ARMS}
    weights = np.full((len(prev.SEEDS), len(x), len(ego), len(model.EXPERTS)), np.nan, np.float32)
    sites, fits = sorted(set(d["sites"])), []
    for k, seed in enumerate(prev.SEEDS):
        for site in sites:
            train = np.flatnonzero(d["sites"] != site)
            held = np.flatnonzero(d["sites"] == site)
            groups = [np.flatnonzero(d["sites"][train] == other) for other in sites if other != site]
            draws = []
            for j, (arm, features) in enumerate(x.items()):
                identity = dict(seed=seed, held=str(site), arm=arm, registration=run.sha(REG),
                    train_ids=base.fit_source.source.array_hash(d["ids"][train]))
                net, checkpoint, used, trace = fit(features, delta, floor, target, valid, scale,
                    train, groups, identity, beat, resume)
                draws.append(used)
                with torch.inference_mode():
                    weights[k, j, held] = net(features[held]).numpy()
                fits.append(dict(identity=identity, checkpoint=checkpoint, held_rows=len(held), trace=trace))
            for used in draws[1:]:
                base.fit_source.source.exact(draws[0], used)
    if len(fits) != 81 or not np.isfinite(weights).all():
        raise ValueError("Incomplete forecasts")
    np.testing.assert_allclose(weights.sum(-1), 1., atol=1e-6)
    path = PRIVATE / "weights.npz"
    tmp = path.with_suffix(".pending.npz")
    np.savez(tmp, ids=d["ids"], weights=weights)
    tmp.replace(path)
    receipt = PRIVATE / "freeze.json"
    run.once(receipt, run.stable(dict(fits=fits, weights=run.artifact(path), registration=run.sha(REG),
        held_source=True, matched_draws=True, past_only=True)))
    return receipt


def score(jobs, cut, frozen, beat):
    if frozen["registration"] != run.sha(REG) or not all(
            frozen[k] for k in ("held_source", "matched_draws", "past_only")):
        raise ValueError("Require complete frozen forecasts before readout")
    d = parent.data(jobs, beat)
    _, _, deltas, _, paths = model.design(d["geometry"][:, :16].reshape(-1, 8, 2))
    queries = parent.old.fitting_layout(jobs[0]["identity"], d)
    with np.load(run.checked(frozen["weights"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], d["ids"])
        coefficients = z["weights"].astype(np.float64)
    y, valid = d["target"], d["valid"]
    clean = np.where(valid[..., None], y, 0.)
    known, endpoint = valid.any(1), valid[:, -1]
    sites, src = np.unique(d["sites"], return_inverse=True)
    w = np.zeros(len(src))
    for q in queries:
        w[q] = 1. / len(q)

    def errors(p):
        dist = np.linalg.norm(p - clean, axis=2)
        return np.where(valid, dist, 0.).sum(1) / valid.sum(1).clip(1), dist[:, -1]

    def mass(v):
        return np.bincount(src, weights=w * v, minlength=len(sites))

    cv, _ = errors(paths["cv"])
    easy = known & (cv > 0) & (cv <= cut["easy"])
    hard = known & (cv >= cut["hard"])
    cv_mass = mass(cv)
    if (cv_mass <= 0).any():
        raise ValueError("Missing source denominator")
    contrasts, absolute, mix, risk = {}, {}, {}, {}
    for k, seed in enumerate(prev.SEEDS):
        pred = dict(paths)
        for j, arm in enumerate(model.ARMS):
            pred[arm] = paths["ols4"] + np.einsum("nk,nkhd->nhd", coefficients[k, j], deltas)
            change = np.linalg.norm(pred[arm] - paths["ols4"], axis=2)
            mix.setdefault(arm, []).append(dict(seed=seed, mean_weights=coefficients[k, j].mean(0),
                median_endpoint_change=float(np.median(change[:, -1]))))
        for a, b in PAIRS:
            values = scores(pred[a], pred[b], y, valid)
            contrasts.setdefault(f"{a}_vs_{b}", []).append({m: 100. * mass(v) / cv_mass
                for m, v in values.items()})
        for a, p in pred.items():
            ade, end = errors(p)
            absolute.setdefault(a, []).append(dict(ade=mass(ade) / mass(known),
                fde=mass(endpoint * end) / mass(endpoint)))
            easy_denom, hard_denom = mass(easy * cv), mass(hard * cv)
            easy_mass = mass(easy * np.maximum(ade - cv, 0.))
            hard_mass = mass(hard * (cv - ade))
            risk.setdefault(a, []).append(dict(seed=seed,
                easy_positive_harm_percent={s: float(100*easy_mass[i]/easy_denom[i]) if easy_denom[i]>0
                    else None for i, s in enumerate(sites)},
                hard_known_gain_percent={s: float(100*hard_mass[i]/hard_denom[i]) if hard_denom[i]>0
                    else None for i, s in enumerate(sites)}))
        beat(state="scored", seed=seed, rows=len(src))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))

    def aggregate(rows):
        return {a: {m: parent.old.metric.intervals(np.stack([r[m] for r in records], 1), draws)
                    for m in records[0]} for a, records in rows.items()}

    return dict(contrasts=aggregate(contrasts), absolute=aggregate(absolute), risk_by_source_seed=risk,
        mixture=mix, experts=model.EXPERTS, rows=len(src), sources=sites, queries=len(queries),
        known_rows=int(known.sum()), full12_rows=int(valid.all(1).sum()), fits=len(frozen["fits"]),
        seeds=prev.SEEDS, development_targets=False, external_access=False,
        independent_confirmation=False, original_rule_unchanged=True, deployed=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "train", "score", "replay"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cut, jobs = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit protocol and code before execution")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(CFG["torch_threads"])
        torch.set_num_interop_threads(1)

        def beat(**values):
            status = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic() - start, **values)
            base.fit_source.source.atomic(PRIVATE / "heartbeat.json", status)
            print(status, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or status["seconds"] > 43200:
                raise OSError("Resource guard; checkpoints are preserved")

        mark = PUBLIC / f"{a.phase}_execution.json"
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            out = train_folds(jobs, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            out = PUBLIC / "analysis.json"
            run.once(out, run.stable(score(jobs, cut, prev.receipt(PUBLIC, "train"), beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic() - start, artifact=run.artifact(out)))


if __name__ == "__main__":
    main()
