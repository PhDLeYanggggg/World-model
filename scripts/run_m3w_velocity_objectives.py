"""Fitting-source loss ablation with frozen, aligned velocity forecasts."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_local_velocity as parent
from src.world_model import m3w_velocity_objectives as model

np, torch, run, prev = parent.np, parent.torch, parent.run, parent.prev
base, cfg = parent.base, parent.CFG
PUBLIC = parent.PUBLIC.parent / "native_velocity_objectives_v1"
PRIVATE = parent.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
FILES = (Path(__file__), Path(model.__file__), ROOT/"tests/test_m3w_velocity_objectives.py", PUBLIC/"protocol.md")
CONTRASTS = (
    ("grid_harm", "ols4"), ("grid_plain", "ols4"), ("masked_harm", "ols4"), ("masked_plain", "ols4"),
    ("grid_plain", "grid_harm"), ("masked_plain", "masked_harm"),
    ("masked_harm", "grid_harm"), ("masked_plain", "grid_plain"),
    ("masked_plain", "fixed_neighbor"), ("fixed_neighbor", "ols4"), ("ols4", "cv"),
)


def register():
    cut, jobs = parent.register()
    prev.receipt(parent.PUBLIC, "replay")
    frozen = prev.receipt(parent.PUBLIC, "train")
    if frozen["registration"] != run.sha(parent.REG) or not all(frozen[k] for k in ("matched_draws", "past_only", "held_source")):
        raise ValueError("Expected verified parent predictions")
    files = (*FILES, Path(parent.__file__), parent.REG, parent.PUBLIC/"train_execution.json",
             parent.PUBLIC/"replay_execution.json")
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in files], settings=cfg, seeds=prev.SEEDS,
        contrast=CONTRASTS, four_cells=True, cached_arm="grid_harm", fresh_arms=model.ARMS,
        bootstrap=3000, diagnostic_complete_label_subset=True, held_fitting_sources=True,
        official_rule_unchanged=True, development_targets=False, external_access=False)))
    return cut, jobs, frozen


def fit(x, y, floor, valid, ix, groups, identity, beat, resume):
    seed = identity["seed"]
    torch.manual_seed(seed)
    net = parent.model.VelocityModel()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, draws, trace = 0, np.zeros(len(ix), np.int64), []
    path = PRIVATE/f'{identity["held"]}_{seed}_{identity["arm"]}.pt'
    if path.exists():
        if not resume:
            raise FileExistsError("Resume required to use existing checkpoints")
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["identity"] != identity or not 0 < st["step"] <= cfg["steps"]:
            raise ValueError("Incompatible checkpoint identity")
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        rng.set_state(st["rng"])
        step, draws, trace = (st[k] for k in ("step", "draws", "trace"))
    while step < cfg["steps"]:
        i = base.native.draw_batch(groups, cfg["batch"], rng)
        at = ix[i]
        delta = net(*(p[at] for p in x))
        loss = model.objective(floor[at]+delta, floor[at], y[at], valid[at], x[-1][at], identity["arm"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ablation training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        step += 1
        np.add.at(draws, i, 1)
        if step % cfg["checkpoint"] == 0:
            trace.append(dict(step=step, loss=float(loss.detach())))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, model=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.get_state(), draws=draws, trace=trace), tmp)
            tmp.replace(path)
            beat(state="checkpoint", **identity, **trace[-1])
    if draws.sum() != cfg["batch"]*cfg["steps"]:
        raise ValueError("Incomplete update budget")
    return net.eval(), run.artifact(path), (draws, rng.get_state())


def train(jobs, frozen, beat, resume):
    d = parent.parent.data(jobs, beat)
    ego, neighbors, masks, basis, scale, paths = parent.model.inputs(d["geometry"])
    x = tuple(map(torch.from_numpy, (ego, neighbors, masks["near"], basis, scale)))
    y, floor, valid = map(torch.from_numpy, (d["target"], paths["ols4"].astype(np.float32), d["valid"]))
    baseline = {(r["identity"]["held"], r["identity"]["seed"]): r for r in frozen["fits"] if r["identity"]["arm"] == "near"}
    sites = sorted(set(d["sites"]))
    output = np.full((len(prev.SEEDS), len(model.ARMS), len(y), 12, 2), np.nan, dtype=np.float32)
    fits = []
    for k, seed in enumerate(prev.SEEDS):
        for site in sites:
            ix, held = (np.flatnonzero(c) for c in (d["sites"] != site, d["sites"] == site))
            groups = [np.flatnonzero(d["sites"][ix] == g) for g in sites if g != site]
            control = baseline[(str(site), seed)]
            st = torch.load(run.checked(control["checkpoint"]), map_location="cpu", weights_only=False)
            train_ids = base.fit_source.source.array_hash(d["ids"][ix])
            if st["identity"] != control["identity"] or st["identity"]["train_ids"] != train_ids or st["step"] != cfg["steps"]:
                raise ValueError("Reference fit lineage or steps changed")
            for a, arm in enumerate(model.ARMS):
                identity = dict(registration=run.sha(REG), held=str(site), seed=seed, arm=arm, train_ids=train_ids)
                net, ck, state = fit(x, y, floor, valid, ix, groups, identity, beat, resume)
                base.fit_source.source.exact(state, (st["draws"], st["rng"]))
                with torch.inference_mode():
                    for h in np.array_split(held, max(1, (len(held)+1023)//1024)):
                        output[k, a, h] = net(*(p[h] for p in x)).numpy()
                fits.append(dict(identity=identity, checkpoint=ck, held_rows=len(held)))
    if len(fits) != 81 or not np.isfinite(output).all():
        raise ValueError("Incomplete ablation prediction bank")
    path = PRIVATE/"predictions.npz"
    tmp = path.with_suffix(".pending.npz")
    np.savez(tmp, ids=d["ids"], delta=output)
    tmp.replace(path)
    receipt = PRIVATE/"freeze.json"
    run.once(receipt, run.stable(dict(registration=run.sha(REG), fits=fits, predictions=run.artifact(path),
        parent_predictions=frozen["predictions"], same_draws=True, past_only=True)))
    return receipt


def score(jobs, cut, frozen, beat):
    if frozen["registration"] != run.sha(REG) or not frozen["same_draws"] or not frozen["past_only"]:
        raise ValueError("Expected frozen past-only ablation")
    d = parent.parent.data(jobs, beat)
    queries = parent.parent.old.fitting_layout(jobs[0]["identity"], d)
    paths0 = parent.model.inputs(d["geometry"])[-1]
    known = d["valid"].any(1)
    full = d["valid"].all(1)
    target = np.where(d["valid"][..., None], d["target"], 0.)
    _, src = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(src))
    for q in queries:
        w[q] = 1/len(q)
    mass = lambda z: np.bincount(src, weights=w*z, minlength=9)
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))

    def ade(p):
        return np.where(d["valid"], np.linalg.norm(p-target, axis=-1), 0.).sum(1) / d["valid"].sum(1).clip(1)

    cv = ade(paths0["cv"])
    easy = known & (cv > 0.) & (cv <= cut["easy"])
    all_mass, full_mass = mass(cv), mass(full*cv)
    if (all_mass <= 0).any() or (full_mass <= 0).any():
        raise ValueError("Fitting sources need observed and complete-label CV support")
    with np.load(run.checked(frozen["predictions"]), allow_pickle=False) as z, np.load(run.checked(frozen["parent_predictions"]), allow_pickle=False) as b:
        for bank in (z, b):
            np.testing.assert_array_equal(d["ids"], bank["ids"])
        change, original = z["delta"], b["delta"][:, parent.model.ARMS.index("near")]
    contrasts, endpoints, risks = {}, {}, {}
    for k, seed in enumerate(prev.SEEDS):
        paths = dict(paths0, grid_harm=paths0["ols4"]+original[k])
        paths.update({a: paths0["ols4"]+change[k, j] for j, a in enumerate(model.ARMS)})
        errors = {a: ade(p) for a, p in paths.items()}
        for a, b in CONTRASTS:
            m = parent.model.scores(paths[a], paths[b], target, d["valid"])
            result = {n: 100*mass(v)/all_mass for n, v in m.items()}
            result["complete_label_gain"] = 100*mass(full*m["observed"])/full_mass
            contrasts.setdefault(f"{a}_vs_{b}", []).append(result)
        for a, p in paths.items():
            move = float(np.median(np.linalg.norm(p[:, -1]-paths0["ols4"][:, -1], axis=1)))
            endpoints.setdefault(a, []).append(move)
            support = mass(easy*cv)
            ratio = np.divide(100*mass(easy*np.maximum(errors[a]-cv, 0.)), support,
                              out=np.full(len(support), np.nan), where=support>0)
            risks.setdefault(a, []).append(dict(max_supported=float(np.nanmax(ratio)),
                                              undefined=int((~np.isfinite(ratio)).sum())))
        beat(state="scored_frozen_seed", seed=seed)
    ci = parent.parent.old.metric.intervals
    return dict(contrasts={a: {m: ci(np.stack([r[m] for r in rows], 1), draws)
                              for m in rows[0]} for a, rows in contrasts.items()},
        endpoint_change_pixels=endpoints, known_easy_harm_diagnostic=risks, rows=len(src),
        complete_rows=int(full.sum()), source_groups=9, queries=len(queries), fresh_fits=len(frozen["fits"]),
        cached_verified_fits=27, seeds=prev.SEEDS, original_rule_unchanged=True, deployment=False,
        development_targets=False, external_access=False, independent_confirmation=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "train", "score", "replay"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cut, jobs, frozen = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit registration before execution")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(cfg["torch_threads"])
        torch.set_num_interop_threads(1)

        def beat(**fields):
            payload = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-t, **fields)
            base.fit_source.source.atomic(PRIVATE/"heartbeat.json", payload)
            if fields.get("step", cfg["steps"]) == cfg["steps"]:
                print(payload, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-t > 43200:
                raise OSError("Resource limit; atomic checkpoints preserved")

        mark = PUBLIC/f"{a.phase}_execution.json"
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            result = train(jobs, frozen, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            result = PUBLIC/"analysis.json"
            run.once(result, run.stable(score(jobs, cut, prev.receipt(PUBLIC, "train"), beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-t, artifact=run.artifact(result)))


if __name__ == "__main__":
    main()
