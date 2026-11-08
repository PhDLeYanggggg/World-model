"""Registered held-source local velocity forecasting experiment."""
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
from src.world_model import m3w_local_velocity as model

base, run, prev, np, torch = parent.base, parent.run, parent.prev, parent.np, parent.torch
PUBLIC = parent.PUBLIC.parent / "native_local_velocity_v1"
PRIVATE = parent.PRIVATE.parent / PUBLIC.name
ORIGINAL_REG = PUBLIC / "registration.json"
REG = PUBLIC / "registration_v2.json"
FILES = (Path(__file__), Path(model.__file__), PUBLIC / "protocol.md",
         ROOT / "tests/test_m3w_local_velocity.py")
CFG = dict(batch=128, steps=800, checkpoint=100, lr=.001, torch_threads=4, workers=0)
CONTRASTS = (("near", "ols4"), ("near", "cv"), ("near", "ego"), ("near", "outer"),
             ("near", "fixed_neighbor"), ("ego", "ols4"), ("outer", "ols4"),
             ("fixed_neighbor", "ols4"), ("ols4", "cv"))


def register():
    _, jobs = parent.register()
    cut = parent.old.register()[0]
    prior = PUBLIC.parent / "native_ols_anchor_retrieval_v1"
    prev.receipt(prior, "replay")
    bound = (*FILES, parent.REG, Path(parent.__file__), prior / "replay_execution.json", ORIGINAL_REG)
    run.once(REG, run.stable(dict(files=[run.artifact(f) for f in bound],
        cfg=CFG, seeds=prev.SEEDS, fit_arms=model.ARMS, contrasts=CONTRASTS, bootstrap=3000,
        primary=["near_vs_ols4", "near_vs_ego", "near_vs_outer"], context="disjoint matched halves of nearest8",
        role="leave_one_fitting_source_out", external_access=False, development_targets=False,
        no_metric_or_seconds=True, official_cv_rule_unchanged=True,
        amendment="Fix three-arm draw check before scoring; retain original checkpoint identity.")))
    return cut, jobs


def fit(x, target, floor, valid, train, groups, identity, beat, resume):
    seed = identity["seed"]
    torch.manual_seed(seed)
    net = model.VelocityModel()
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, draws, trace = 0, np.zeros(len(train), np.int64), []
    path = PRIVATE / f'{identity["held"]}_{seed}_{identity["arm"]}.pt'
    if path.exists():
        if not resume:
            raise FileExistsError("Existing checkpoints require --resume")
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["identity"] != identity or not 0 < st["step"] <= CFG["steps"]:
            raise ValueError("Resume identity changed")
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        rng.set_state(st["rng"])
        step, draws, trace = (st[k] for k in ("step", "draws", "trace"))
    while step < CFG["steps"]:
        i = base.native.draw_batch(groups, CFG["batch"], rng)
        at = train[i]
        y = net(*(p[at] for p in x))
        d = model.lower(floor[at]+y, floor[at], target[at], valid[at]) / (12*x[-1][at])
        loss = (d+d.relu()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite velocity loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        step += 1
        np.add.at(draws, i, 1)
        if step % CFG["checkpoint"] == 0:
            trace.append(dict(step=step, loss=float(loss.detach())))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, model=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.get_state(), draws=draws, trace=trace), tmp)
            tmp.replace(path)
            beat(state="fit_checkpoint", seed=seed, arm=identity["arm"], held=identity["held"], **trace[-1])
    if draws.sum() != CFG["steps"]*CFG["batch"]:
        raise ValueError("Draw budget changed")
    return net.eval(), run.artifact(path), (draws, rng.get_state())


def train(jobs, beat, resume):
    d = parent.data(jobs, beat)
    ego, neighbors, masks, basis, scale, paths = model.inputs(d["geometry"])
    np.testing.assert_allclose(paths["cv"], d["geometry"][:, 332:356].reshape(-1, 12, 2), atol=.0001)
    y, f, valid = map(torch.from_numpy, (d["target"], paths["ols4"].astype(np.float32), d["valid"]))
    output = np.full((len(prev.SEEDS), len(model.ARMS), len(y), 12, 2), np.nan, dtype=np.float32)
    sites = sorted(set(d["sites"]))
    fits = []
    for k, seed in enumerate(prev.SEEDS):
        for held in sites:
            ix, out = (np.flatnonzero(v) for v in (d["sites"] != held, d["sites"] == held))
            groups = [np.flatnonzero(d["sites"][ix] == s) for s in sites if s != held]
            stats = []
            for j, arm in enumerate(model.ARMS):
                x = tuple(map(torch.from_numpy, (ego, neighbors, masks[arm], basis, scale)))
                identity = dict(seed=seed, arm=arm, held=str(held), registration=run.sha(ORIGINAL_REG),
                    train_ids=base.fit_source.source.array_hash(d["ids"][ix]))
                net, ck, state = fit(x, y, f, valid, ix, groups, identity, beat, resume)
                stats.append(state)
                with torch.inference_mode():
                    for at in np.array_split(out, max(1, (len(out)+1023)//1024)):
                        output[k, j, at] = net(*(a[at] for a in x)).numpy()
                fits.append(dict(identity=identity, checkpoint=ck, held_rows=len(out)))
            for state in stats[1:]:
                base.fit_source.source.exact(stats[0], state)
    if not np.isfinite(output).all() or len(fits) != 81:
        raise ValueError("Expected complete frozen held-source predictions")
    path = PRIVATE / "predictions.npz"
    tmp = path.with_suffix(".pending.npz")
    np.savez(tmp, ids=d["ids"], delta=output)
    tmp.replace(path)
    receipt = PRIVATE / "freeze.json"
    run.once(receipt, run.stable(dict(fits=fits, predictions=run.artifact(path), registration=run.sha(REG),
        held_source=True, matched_draws=True, past_only=True)))
    return receipt


def score(jobs, cut, frozen, beat):
    if frozen["registration"] != run.sha(REG) or not all(frozen[k] for k in ("held_source", "matched_draws", "past_only")):
        raise ValueError("Require registered past-only held-source predictions")
    d = parent.data(jobs, beat)
    queries = parent.old.fitting_layout(jobs[0]["identity"], d)
    _, _, masks, _, _, base_paths = model.inputs(d["geometry"])
    with np.load(run.checked(frozen["predictions"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], d["ids"])
        delta = z["delta"]
    target, valid = d["target"], d["valid"]
    known = valid.any(1)
    clean = np.where(valid[..., None], target, 0.)
    groups, src = np.unique(d["sites"], return_inverse=True)
    w = np.zeros(len(src))
    for q in queries:
        w[q] = 1./len(q)

    def ade(p):
        dist = np.linalg.norm(p-clean, axis=2)
        return np.where(valid, dist, 0.).sum(1) / valid.sum(1).clip(1), dist[:, -1]

    cv = ade(base_paths["cv"])[0]
    easy = known & (cv > 0) & (cv <= cut["easy"])
    hard = known & (cv >= cut["hard"])
    mass = lambda value: np.bincount(src, weights=w*value, minlength=len(groups))
    cv_mass = mass(cv)
    if (cv_mass <= 0).any():
        raise ValueError("CV denominator missing on a registered source")
    draws = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    contrasts, absolute, context = {}, {}, {}
    for k, seed in enumerate(prev.SEEDS):
        paths = dict(base_paths)
        paths.update({a: base_paths["ols4"]+delta[k, j] for j, a in enumerate(model.ARMS)})
        errors = {a: ade(p) for a, p in paths.items()}
        for a, b in CONTRASTS:
            all_metrics = {name: mass(g)/cv_mass*100
                           for name, g in model.scores(paths[a], paths[b], target, valid).items()}
            contrasts.setdefault(f"{a}_vs_{b}", []).append(all_metrics)
        for a, (err, end) in errors.items():
            positive_easy_harm = mass(easy*np.maximum(err-cv, 0.))
            easy_mass = mass(easy*cv)
            risk = np.divide(100*positive_easy_harm, easy_mass, out=np.full(len(groups), np.nan), where=easy_mass>0)
            hard_mass = mass(hard*cv)
            hard_gain = np.divide(100*mass(hard*(cv-err)), hard_mass, out=np.full(len(groups), np.nan), where=hard_mass>0)
            m = dict(ade=mass(err)/mass(known), fde=mass(valid[:, -1]*end)/mass(valid[:, -1]),
                     max_easy_positive_harm=max((float(x) for x in risk if np.isfinite(x)), default=None),
                     undefined_easy=int((~np.isfinite(risk)).sum()),
                     hard_gain_known=(float(hard_gain.mean()) if np.isfinite(hard_gain).all() else None))
            absolute.setdefault(a, []).append(m)
        for a in model.ARMS:
            dist = np.linalg.norm(paths[a]-base_paths["ols4"], axis=2)
            context.setdefault(a, []).append(dict(changed_rows=int((dist.max(1)>1e-6).sum()),
                                                 median_endpoint_delta=float(np.median(dist[:, -1]))))
        beat(state="scored_frozen_seed", seed=seed, rows=len(src))
    ci = lambda x: parent.old.metric.intervals(np.stack(x, 1), draws)
    res = {}
    for key, rows in contrasts.items():
        res[key] = {m:ci([r[m] for r in rows]) for m in rows[0]}
    summary = {}
    for key, rows in absolute.items():
        summary[key] = dict(**{m:ci([r[m] for r in rows]) for m in ("ade", "fde")},
                            easy_positive_harm_max_by_seed=[r["max_easy_positive_harm"] for r in rows],
                            undefined_easy_sources_by_seed=[r["undefined_easy"] for r in rows],
                            hard_gain_known_by_seed=[r["hard_gain_known"] for r in rows])
    return dict(contrasts=res, absolute=summary, context=context, rows=len(src), source_groups=len(groups),
        known_rows=int(known.sum()), whole12_rows=int(valid.all(1).sum()), paired_context_rows=int(masks["near"].any(1).sum()),
        queries=len(queries), fits=len(frozen["fits"]), seeds=prev.SEEDS,
        development_targets=False, external_access=False, independent_confirmation=False,
        official_cv_rule_unchanged=True, deployed=False, stage5c_executed=False, smc_enabled=False)


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
            raise ValueError("Commit the experimental registration before running")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(CFG["torch_threads"])
        torch.set_num_interop_threads(1)

        def beat(**values):
            item = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **values)
            base.fit_source.source.atomic(PRIVATE / "heartbeat.json", item)
            print(item, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError("Resource guard stopped with checkpoints preserved")

        mark = PUBLIC / f"{a.phase}_execution.json"
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            out = train(jobs, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            out = PUBLIC / "analysis.json"
            run.once(out, run.stable(score(jobs, cut, prev.receipt(PUBLIC, "train"), beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))


if __name__ == "__main__":
    main()
