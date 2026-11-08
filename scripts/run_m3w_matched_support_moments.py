"""Train three matched seeds per held group; compare at identical recording totals."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use the native arm64 .venv-pytorch environment")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recipient_release_crossfit as old
from src.world_model import m3w_matched_support_moments as model
from src.evaluation.m3w_selection_forecast_readout import errors


run, np, cap, prior, torch = old.run, old.np, old.cap, old.prior, old.prior.base.torch
OUT = cap.OUT.parent/"matched_support_moments_v1"
PRIVATE = cap.PRIVATE.parent/OUT.name
SEEDS, STEPS, BATCH, EVERY = (17, 29, 43), 500, 96, 100
HEADS = ("marginal", "supervised")
ARMS = ("marginal_gain", "marginal_record", "supervised_gain", "supervised_record", "supervised_query")
PAIRS = {f"{a}_vs_{'cv' if b == prior.cal.BASE else b}": (b, a, prior.cal.BASE)
         for a, b in ((prior.cal.OLS, prior.cal.BASE),
                      *((a, b) for a in ARMS for b in (prior.cal.BASE, prior.cal.OLS)),
                      ("supervised_record", "marginal_record"), ("supervised_record", "supervised_gain"),
                      ("supervised_record", "supervised_query"), ("marginal_record", "marginal_gain"))}


def register():
    parent, identity, cut = old.register()
    if run.read(cap.OUT/"controls.json")["summary"]["curved_total"]["verified_both"] != 14:
        raise ValueError("Require the fixed recording-total capacity reference")
    path = OUT/"registration.json"
    run.once(path, run.stable(dict(code=run.artifact(Path(__file__)), head=run.artifact(Path(model.__file__)),
        input_source=run.artifact(Path(old.__file__)), prior=parent,
        budget_reference=run.artifact(cap.OUT/"controls.json"), seeds=SEEDS,
        steps=STEPS, batch_size=BATCH, checkpoint_every=EVERY, workers=0, cpu_threads=4,
        lr=.001, primary="supervised_record_vs_marginal_record_direct_conservative",
        controls=ARMS, risk_budget=.02, forced_counts_are_diagnostic_not_deployment=True,
        held_groups="nine leave-source-out folds with fold-local training easy cut",
        calibration_access=False, confirmation_access=False, risk_rule_changed=False)))
    return run.sha(path), identity, cut


def prepare(identity, beat):
    inputs, targets, blocks, offset = [], [], [], 0
    for rec, g, h, cv, ols, bank, y, valid, inv, counts, w in old.blocks(identity, beat):
        x, scale = old.features(h, bank, cv, inv)
        raw = errors(cv, y, valid)[0]
        known = valid.any(1)
        truth = np.stack([errors(p, y, valid)[0] for p in bank.swapaxes(0, 1)], axis=1)
        gain = np.where(known[:, None], raw[:, None]-truth, 0.)/scale[:, None]
        radius = np.linalg.norm(bank-cv[:, None], axis=-1).max(2)/scale[:, None]
        if (np.abs(gain) > radius+1e-8).any():
            raise ValueError("Gain must fit the deterministic displacement envelope")
        inputs.append((x.astype(np.float32), radius.astype(np.float32)))
        targets.append((np.where(known, raw/scale, 0.).astype(np.float32), gain.astype(np.float32), known,
                        raw.astype(float)))
        blocks.append(dict(rec=rec, source=g, sl=slice(offset, offset+len(cv)), cv=cv, ols=ols, bank=bank,
                           target=y, valid=valid, inverse=inv, counts=counts, weight=w, scale=scale))
        offset += len(cv)
    join = lambda cells: tuple(np.concatenate(v) for v in zip(*cells, strict=True))
    if len(blocks) != 24 or offset != 69746 or sum(int(b["counts"].sum()) for b in blocks) != 31923:
        raise ValueError("Fixed fitting population changed")
    return tuple(torch.from_numpy(a) for a in join(inputs)), join(targets), blocks


def group_ids(blocks):
    return {g: np.concatenate([np.arange(b["sl"].start, b["sl"].stop) for b in blocks if b["source"] == g])
            for g in sorted({b["source"] for b in blocks})}


def fit_one(binding, head, held, seed, groups, x, target, beat, replay):
    p = PRIVATE/f"fit_{held}_{seed}_{head}.pt"
    ids = [v for k, v in groups.items() if k != held]
    train = np.concatenate(ids)
    b, gain, known, raw = target
    support = raw[train][np.isfinite(raw[train]) & (raw[train] > 0)]
    cut = float(np.quantile(support, .25))
    easy = torch.from_numpy(known & (raw > 0.) & (raw <= cut))
    b, gain, known = map(torch.from_numpy, (b, gain, known))
    shape = x[0].shape[2]
    identity = dict(registration=binding, arm=head, held=held, seed=seed,
                    training_sources=[k for k in groups if k != held], easy_cut=cut)
    torch.manual_seed(seed)
    net = model.SupportMoments(shape)
    initial = hashlib.sha256(b"".join(a.detach().numpy().tobytes() for a in net.parameters())).hexdigest()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, trace, draws = 0, [], "0"*64
    if p.exists():
        s = torch.load(p, map_location="cpu", weights_only=True)
        if s["identity"] != identity or s["initial"] != initial or not 0 < s["step"] <= STEPS:
            raise ValueError("Fit checkpoint identity changed")
        net.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        rng.set_state(s["rng"])
        step, trace, draws = s["step"], s["trace"], s["draws"]
    if replay and step != STEPS:
        raise ValueError("Replay may not train")
    for i in range(step, STEPS):
        ix = prior.base.fit.draw_batch(ids, BATCH, rng)
        pred = net(x[0][ix], x[1][ix])
        loss = model.objective(pred, b[ix], gain[ix], known[ix], easy[ix], x[1][ix], head == "supervised")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite risk training objective")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        draws = hashlib.sha256(bytes.fromhex(draws)+ix.tobytes()).hexdigest()
        if (i+1) % EVERY == 0:
            trace.append(float(loss.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=identity, initial=initial, model=net.state_dict(), optimizer=opt.state_dict(),
                step=i+1, trace=trace, rng=rng.get_state(), draws=draws), tmp)
            tmp.replace(p)
            beat(stage="checkpoint", held=held, arm=head, seed=seed, step=i+1, loss=trace[-1])
    return net.eval(), dict(checkpoint=run.artifact(p), identity=identity, initial=initial, draws=draws)


@torch.inference_mode()
def predict(net, block, x):
    sl = block["sl"]
    pred = [net(a, b)[:2] for a, b in zip(x[0][sl].split(1024), x[1][sl].split(1024), strict=True)]
    ref, cost = (torch.cat(v).numpy().astype(float) for v in zip(*pred, strict=True))
    return ref*block["scale"][:, None], cost*block["scale"][:, None, None]


def freeze(binding, data, beat, replay):
    x, labels, blocks = data
    groups = group_ids(blocks)
    if len(groups) != 9:
        raise ValueError("Require nine fitting-source exclusions")
    receipts, fits = [], []
    for g in groups:
        for seed in SEEDS:
            pair = [fit_one(binding, k, g, seed, groups, x, labels, beat, replay) for k in HEADS]
            (_, ma), (_, su) = pair
            if (ma["initial"], ma["draws"], ma["identity"]["easy_cut"]) != (
                    su["initial"], su["draws"], su["identity"]["easy_cut"]):
                raise ValueError("Matched model initialization or sampler differs")
            fits.extend((ma, su))
            for b in blocks:
                if b["source"] != g:
                    continue
                p = PRIVATE/f"choice_{b['rec']}_{seed}.json"
                bind = dict(registration=binding, recording=b["rec"], seed=seed, fits=[ma["checkpoint"], su["checkpoint"]])
                if not p.exists():
                    if replay:
                        raise ValueError("Replay may not produce missing predictions")
                    legal = np.any(b["bank"] != b["cv"][:, None], axis=(2, 3))
                    budget = int(b["counts"].sum())
                    result = []
                    for net, info in pair:
                        if g in info["identity"]["training_sources"]:
                            raise ValueError("Held source used during its fit")
                        ref, cost = predict(net, b, x)
                        args = (ref, cost, legal, b["inverse"], budget, b["weight"])
                        result.extend(model.allocate(*args))
                        if info["identity"]["arm"] == "supervised":
                            result.append(model.allocate(*args, query_counts=b["counts"])[1])
                    run.once(p, run.stable(dict(bind, decisions=[(c.tolist(), r) for c, r in result])))
                r = run.read(p)
                if any(r[k] != v for k, v in bind.items()):
                    raise ValueError("Frozen path selection mismatch")
                receipts.append(run.artifact(p))
    if len(receipts) != 72 or len(fits) != 54:
        raise ValueError("Missing matched training or prediction cells")
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, fits=fits, predictions=receipts,
        calibration_access=False, confirmation_access=False, all_held_choices_frozen_before_scoring=True)))
    return receipts


def score(binding, marks, blocks, cut, beat, replay):
    by_rec = {b["rec"]: b for b in blocks}
    rows, diagnostics = [], []
    for mark in marks:
        d = run.read(run.checked(mark))
        rec, seed = d["recording"], d["seed"]
        b = by_rec[rec]
        p = PRIVATE/f"score_{rec}_{seed}.json"
        if not p.exists():
            if replay:
                raise ValueError("Replay may not reopen score cells")
            paths = {prior.cal.BASE: b["cv"], prior.cal.OLS: b["ols"]}
            predictions = {}
            for arm, (chosen, ratio) in zip(ARMS, d["decisions"], strict=True):
                chosen = np.asarray(chosen)
                on = chosen >= 0
                path = b["cv"].copy()
                path[on] = b["bank"][np.flatnonzero(on), chosen[on]]
                if (np.any(path != b["cv"], axis=(1, 2)).sum() != b["counts"].sum()
                        or (arm == "supervised_query" and not np.array_equal(
                            np.bincount(b["inverse"][on], minlength=len(b["counts"])), b["counts"]))):
                    raise ValueError("Intervention budgets differ")
                paths[arm] = path
                predictions[arm] = ratio
            row = prior.base.score_recording(paths, PAIRS, b["target"], b["valid"], b["inverse"], rec,
                                            easy_cut=cut["easy"], hard_cut=cut["hard"])
            run.once(p, run.stable(dict(forecast=mark, row=dict(row, source=b["source"], seed=seed),
                                       predicted_ratio=predictions)))
        r = run.read(p)
        if r["forecast"] != mark or d["registration"] != binding:
            raise ValueError("Score artifact mismatch")
        rows.append(r["row"])
        diagnostics.append(r["predicted_ratio"])
        if len(rows) % 12 == 0:
            beat(stage="score", completed=len(rows))
    summary = prior.base.grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, SEEDS)
    for k in ARMS:
        if summary["risk"][k]["switches"] != 3*31923:
            raise ValueError("Same exact recording budgets required")
    rates = {k: dict(
        predicted_pass=sum(all(v is not None and v <= .02 for v in d[k]) for d in diagnostics),
        predicted_pass_actual_fail=sum(all(v is not None and v <= .02 for v in d[k])
            and not r["metrics"][k]["risk"]["supported"] for d, r in zip(diagnostics, rows, strict=True)),
        known_all_violations=sum(r["metrics"][k]["risk"]["known_all_risk"] is not None
                                 and r["metrics"][k]["risk"]["known_all_risk"] > .02 for r in rows))
        for k in ARMS}
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, summary=summary,
        diagnostic_risk=rates, fitted_models=54, matched_switches_per_arm=3*31923,
        calibration_access=False, confirmation_access=False, deployment=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    a = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            if v.get("step", STEPS) == STEPS:
                print(v, flush=True)

        data = prepare(identity, beat)
        marks = freeze(binding, data, beat, a.replay)
        score(binding, marks, data[2], cut, beat, a.replay)


if __name__ == "__main__":
    main()
