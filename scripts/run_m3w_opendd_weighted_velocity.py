"""Refit weighted velocity on the existing OpenDD location-held development roster."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_moving_future_support as moving
from scripts import run_m3w_opendd_location as source
from src.evaluation.m3w_recording_forecast_comparison import score_recording
from src.world_model import m3w_ols_replacement as motion

study, job, run, np = moving.study, moving.job, moving.run, moving.np
model, torch = study.model, study.torch
OUT = moving.HOME / "opendd_weighted_velocity_v1"
PRIVATE = moving.PRIVATE.parent / OUT.name
ARMS = ("fixed_robust", "ade", "dual")
STEPS = 600
PAIRS = {f"{a}_vs_{'cv' if b == job.CV else b}": (b, a, job.CV)
         for a, b in ((job.OLS, job.CV), *((a, b) for a in ARMS for b in (job.CV, job.OLS)),
                      ("dual", "ade"), ("dual", "fixed_robust"))}


def register():
    old = source.PUBLIC / "registration.json"
    a = moving.OUT / "analysis.json"
    doc = run.read(old)
    if doc["dronecrowd_confirmation_access"] or run.read(a)["confirmation_access"]:
        raise ValueError("Existing exposed development sources only")
    caches = {}
    for site in doc["sites"]:
        p = source.PRIVATE / f"{site}_cohort.npz"
        receipt = run.read(p.with_suffix(".json"))
        if receipt["registration"] != run.sha(old) or receipt["sha256"] != run.sha(p):
            raise ValueError("Cohort cache failed its original hash receipt")
        caches[site] = run.artifact(p)
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        moving_audit=run.artifact(a), opendd_role=run.artifact(old), caches=caches,
        sites=doc["sites"], stride=doc["stride"], seeds=study.job.SEEDS, steps=STEPS,
        threads=4, workers=0, original_2pct=True, zero_shot=False, confirmation_access=False)))
    return run.sha(f), caches


def load(caches):
    sets = {}
    for site, ref in caches.items():
        with np.load(run.checked(ref), allow_pickle=False) as f:
            h, y, offsets, recording = (f[k] for k in ("history", "future", "offsets", "recording"))
        sizes = np.diff(offsets)
        q = np.repeat(np.arange(len(sizes)), sizes)
        rec = np.repeat(recording, sizes)
        _, cv, ols, _, scale = motion.features(h)
        valid = np.isfinite(y).all(2)
        h = (h-h[:, -1:]).astype("float32")
        raw = job.errors(cv, y, valid)[0]
        data = tuple(torch.from_numpy(a.astype("float32")) for a in (h, cv, ols, y)) + (
            torch.from_numpy(valid), torch.from_numpy(scale.astype("float32")))
        sets[site] = dict(data=data, cv=cv, ols=ols, y=y, valid=valid, q=q, rec=rec, raw=raw)
    return sets


def fit(key, held, name, seed, sets, beat, replay):
    src = [s for s in sorted(sets) if s != held]
    pool, cursor = [], 0
    for s in src:
        n = len(sets[s]["raw"])
        pool.append(np.arange(cursor, cursor+n))
        cursor += n
    data = list(map(torch.cat, zip(*(sets[s]["data"] for s in src), strict=True)))
    errors = np.concatenate([sets[s]["raw"] for s in src])
    good = errors[np.isfinite(errors) & (errors > 0.)]
    cut, hard = np.quantile(good, [.25, .75]).tolist()
    easy = torch.from_numpy(np.isfinite(errors) & (errors > 0.) & (errors <= cut))
    identity = dict(key=key, held=held, fit=src, arm=name, seed=seed, easy_cut=cut, hard_cut=hard)
    f = PRIVATE / f"{held}_{seed}_{name}.pt"
    torch.manual_seed(seed)
    net = model.WeightedVelocity()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, trace = 0, []
    if f.exists():
        s = torch.load(f, weights_only=True, map_location="cpu")
        if s["identity"] != identity or not 0 < s["step"] <= STEPS:
            raise ValueError("Checkpoint split or training state changed")
        net.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        rng.set_state(s["rng"])
        step, trace = s["step"], s["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay must not train")
    for step in range(step+1, STEPS+1):
        i = study.direction.timing.draw_batch(pool, 128, rng)
        h, cv, ols, y, valid, scale = (z[i] for z in data)
        loss = model.objective(net(h)[0], cv, ols, y, valid, scale, easy[i], name == "dual")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0:
            trace.append(float(loss.detach()))
            temp = f.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, trace=trace, model=net.state_dict(),
                optimizer=opt.state_dict(), rng=rng.get_state()), temp)
            temp.replace(f)
            beat(held=held, name=name, seed=seed, step=step, loss=trace[-1])
    return net.eval(), cut, hard, run.artifact(f)


def execute(key, caches, beat, replay):
    sets = load(caches)
    rows, roster, models, support = [], {}, [], []
    for site, d in sets.items():
        h = d["data"][0]
        valid = d["valid"]
        support.append(dict(site=site, rows=len(h), requested=int(valid.size), known=int(valid.sum()),
            complete=int(valid.all(1).sum()), wholly_missing=int((~valid.any(1)).sum())))
        robust = model.robust_path(h.to(torch.float64)).numpy()
        for seed in job.SEEDS:
            heads = {a: fit(key, site, a, seed, sets, beat, replay) for a in ARMS[1:]}
            cut, hard = heads["ade"][1:3]
            if heads["dual"][1:3] != (cut, hard):
                raise ValueError("Mismatch in fit-only risk cutoffs")
            models.extend(v[3] for v in heads.values())
            with torch.inference_mode():
                paths = {a: net(h)[0].numpy().astype(float) for a, (net, *_) in heads.items()}
            paths.update({job.CV: d["cv"], job.OLS: d["ols"], "fixed_robust": robust})
            for rec in sorted(set(d["rec"])):
                ix = d["rec"] == rec
                roster[rec] = site
                row = score_recording({k: v[ix] for k, v in paths.items()}, PAIRS, d["y"][ix], valid[ix],
                    d["q"][ix], rec, easy_cut=cut, hard_cut=hard)
                rows.append(dict(row, source=site, seed=seed))
    summary = job.grouped_readout(rows, roster, job.SEEDS)
    refs = summary["effects"][f"{job.OLS}_vs_cv"]
    debt = refs["known_pp"]["point"]-refs["completion_lower_pp"]["point"]
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, models=models, support=support,
        total_rows=sum(r["rows"] for r in support), summary=summary, reference_debit_floor_pp=debt,
        zero_shot=False, roles_unchanged=True, official_calibration_access=False,
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
        key, caches = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            if v["step"] == STEPS:
                print(v, flush=True)

        execute(key, caches, beat, args.replay)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
