"""Fit matched direction heads on disjoint development sources."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_query_temporal_envelope as timing
from src.world_model import m3w_matched_direction as model

job, run, np, torch = timing.job, timing.run, timing.np, timing.torch
OUT = timing.OUT.parent / "matched_direction_v2"
PRIVATE = timing.PRIVATE.parent / OUT.name
ARMS = ("flat", "ego", "neighbors")
POLICIES = (*ARMS, "cal_flat", "cal_ego", "cal_neighbors")
PAIRS = {f"{a}_vs_{'cv' if b == job.CV else b}": (b, a, job.CV)
         for a, b in ((job.OLS, job.CV), *((a, b) for a in POLICIES for b in (job.CV, job.OLS)),
                      ("ego", "flat"), ("neighbors", "flat"), ("neighbors", "ego"))}
SCORE = job.prior.prior.prior.base.score_recording


def register():
    parent, identity, cut, freeze = timing.register()
    capacity = timing.OUT / "capacity.json"
    if not run.read(capacity)["does_not_bound_other_recipients_or_directions"]:
        raise ValueError("Require the timing-only diagnostic")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(parent=parent, earlier_capacity=run.artifact(capacity),
        source_roster=run.artifact(job.OUT / "freeze.json"), code=run.artifact(Path(__file__)),
        model=run.artifact(Path(model.__file__)), steps=timing.STEPS, every=timing.EVERY,
        seeds=job.SEEDS, batch=128, lr=.001, cpu_threads=4, workers=0, original_2pct=True,
        per_step_edit_norm=".25 * frozen nominee minus CV", same_recipients=True,
        neighbor_geometry="first four encoded neighbors in common ego-relative frame and complete past masks",
        primary="neighbors_vs_ego_direct_paired_completion_lower", secondary="both_vs_flat",
        smoothness=.025, max_angle_radians=float(np.pi), calibration_access=False,
        confirmation_access=False, hindsight_direction_used_for_policy=False)))
    return run.sha(f), identity, cut, freeze


def cells(x, blocks, freeze, fold, history, neighbors, masks):
    old, nom = timing.inputs(x, blocks, fold, freeze["registration"])
    result = []
    for b, i, _, d, descriptors, scaled in old:
        s = b["sl"]
        extra = model.project_context(history[s][i], neighbors[s][i], masks[s][i], d, b["scale"][i])
        inputs = torch.cat((descriptors, torch.from_numpy(extra)), dim=1)
        result.append((b, i, .25*d, inputs, .25*scaled))
    return result, nom


def fit(binding, fold, nom, sets, easy_cut, name, beat, replay):
    payload, groups, offset = [], {}, 0
    for b, i, _, features, edit in sets:
        if b["source"] not in fold["fit"] or not len(i):
            continue
        mask = b["valid"][i]
        d = np.where(mask[..., None], b["target"][i]-b["cv"][i], 0.)
        ols = b["ols"][i]-b["cv"][i]
        scale = b["scale"][i, None, None]
        ade = np.linalg.norm(d, axis=2).sum(1)/mask.sum(1).clip(1)
        payload.append((features, edit, torch.from_numpy((d/scale).astype("float32")),
            torch.from_numpy((ols/scale).astype("float32")), torch.from_numpy(mask),
            torch.from_numpy((ade > 0.) & (ade <= easy_cut))))
        groups.setdefault(b["source"], []).append(np.arange(offset, offset+len(i)))
        offset += len(i)
    if set(groups) != set(fold["fit"]):
        raise ValueError("Require all six fitting groups")
    data = list(map(torch.cat, zip(*payload, strict=True)))
    train = [np.concatenate(groups[g]) for g in sorted(groups)]
    seed, held = fold["seed"], fold["held"]
    key = dict(registration=binding, held=held, seed=seed, name=name, nominee=nom,
               fit=fold["fit"], calibration=fold["calibration"], easy_cut=easy_cut)
    p = PRIVATE / f"{name}_{held}_{seed}.pt"
    torch.manual_seed(seed)
    net = model.DirectionHead(data[0].shape[1], name == "neighbors")
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["key"] != key or not 0 < ck["step"] <= timing.STEPS:
            raise ValueError("Checkpoint role or identity mismatch")
        net.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        rng.set_state(ck["rng"])
        step, trace = ck["step"], ck["trace"]
    if replay and step != timing.STEPS:
        raise ValueError("Replay cannot fit")
    for step in range(step+1, timing.STEPS+1):
        ix = timing.draw_batch(train, 128, rng)
        x, edit, target, ols, valid, easy = (z[ix] for z in data)
        prediction, angles = net(x, edit)
        value = timing.model.objective(prediction, angles, target, ols, valid, easy)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % timing.EVERY == 0:
            trace.append(float(value.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(key=key, model=net.state_dict(), optimizer=optimizer.state_dict(),
                            rng=rng.get_state(), step=step, trace=trace), tmp)
            tmp.replace(p)
            beat(stage="checkpoint", name=name, held=held, seed=seed, step=step, loss=trace[-1])
    return net.eval(), run.artifact(p)


@torch.inference_mode()
def forecasts(cell, heads):
    b, i, d, inputs, scaled = cell
    flat = b["cv"].copy()
    flat[i] += d
    out = {"flat": flat}
    for key, head in heads.items():
        path = b["cv"].copy()
        theta = head(inputs, scaled)[1].to(torch.float64)
        path[i] += model.rotate(torch.from_numpy(d), theta).numpy()
        np.testing.assert_allclose(np.linalg.norm(path[i]-b["cv"][i], axis=-1),
                                   np.linalg.norm(d, axis=-1), atol=1e-9, rtol=1e-9)
        out[key] = path
    return out


def hindsight(cell, cut):
    b, i, d, _, _ = cell
    path = b["cv"].copy()
    observed = b["valid"][i]
    unknown = ~observed.any(1)
    target = np.where(unknown[:, None, None], b["ols"][i], b["target"][i])
    # Wholly unknown rows have identical CV debit; align them to OLS for its best lower score.
    path[i] += model.circle_oracle(d, target-b["cv"][i], observed | unknown[:, None])
    np.testing.assert_allclose(np.linalg.norm(path[i]-b["cv"][i], axis=-1),
                               np.linalg.norm(d, axis=-1), atol=1e-9, rtol=1e-9)
    pair = {"oracle_vs_cv": (job.CV, "oracle", job.CV), "oracle_vs_ols": (job.OLS, "oracle", job.CV)}
    row = SCORE({job.CV: b["cv"], job.OLS: b["ols"], "oracle": path}, pair,
        b["target"], b["valid"], b["inverse"], b["rec"], easy_cut=cut["easy"], hard_cut=cut["hard"])
    return dict(row, source=b["source"], seed=job.SEEDS[0])


def execute(binding, identity, cut, freeze, beat, replay):
    x, truth, blocks = job.prior.prior.prepare(identity, beat)
    source = job.prior.prior.prior.base.train_source.fitting_only(identity, beat)
    geometry = source["geometry"]
    for b in blocks:
        np.testing.assert_array_equal(geometry[b["sl"], 332:356], b["cv"].reshape(-1, 24))
        np.testing.assert_array_equal(source["valid"][b["sl"]], b["valid"])
        np.testing.assert_array_equal(source["target"][b["sl"]], b["target"])
    history = geometry[:, :16].reshape(-1, 8, 2).copy()
    neighbors = geometry[:, 38:166].reshape(-1, 8, 8, 2)[:, :4].copy()
    masks = geometry[:, 230:294].reshape(-1, 8, 8)[:, :4].astype(bool)
    del geometry, source
    groups = job.prior.prior.group_ids(blocks)
    fit_rows, rows, capacity = [], [], []
    roles = freeze["calibration"]
    if [(r["held"], r["seed"]) for r in roles] != [(g, s) for g in groups for s in job.SEEDS]:
        raise ValueError("Frozen folds or seed order changed")
    for pos, fold in enumerate(roles):
        held, seed = fold["held"], fold["seed"]
        a, c = job.model.roles(groups, held)
        if (a, c) != (fold["fit"], fold["calibration"]):
            raise ValueError("Fold roles overlap or changed")
        if pos % len(job.SEEDS) == 0:
            sets, nom = cells(x[0], blocks, freeze, fold, history, neighbors, masks)
        raw = truth[-1][np.concatenate([groups[g] for g in a])]
        easy = float(np.quantile(raw[np.isfinite(raw) & (raw > 0.)], .25))
        fitted = {k: fit(binding, fold, nom, sets, easy, k, beat, replay) for k in ARMS[1:]}
        heads = {k: v[0] for k, v in fitted.items()}
        calibration = {k: [] for k in ARMS}
        for cell in sets:
            b, i = cell[:2]
            if b["source"] in c:
                for k, p in forecasts(cell, heads).items():
                    calibration[k].append((b, np.isin(np.arange(len(b["cv"])), i), job.calibration_cell(b, p, cut)))
        trials = {}
        for k, items in calibration.items():
            bb, ok, cc = zip(*items, strict=True)
            trials[k] = job.trial(bb, [np.zeros(len(b["cv"])) for b in bb], list(zip(ok, cc, strict=True)), 0.)
        chosen = {k: timing.select(trials, (k,)) for k in ARMS}
        cal = run.stable(dict(registration=binding, held=held, seed=seed, calibration=c, fit=a, choices=chosen,
                              metrics=trials, fits={k: v[1] for k, v in fitted.items()}))
        path = PRIVATE / f"policy_{held}_{seed}.json"
        if replay and not path.exists():
            raise ValueError("Replay cannot choose a policy")
        run.once(path, cal)
        fit_rows.append(cal)
        for cell in sets:
            b = cell[0]
            if b["source"] != held:
                continue
            p = forecasts(cell, heads)
            p.update({job.CV: b["cv"], job.OLS: b["ols"]})
            for k in ARMS:
                p[f"cal_{k}"] = p[chosen[k] or job.CV]
            row = SCORE(p, PAIRS, b["target"], b["valid"], b["inverse"], b["rec"],
                        easy_cut=cut["easy"], hard_cut=cut["hard"])
            rows.append(dict(row, source=held, seed=seed))
            if seed == job.SEEDS[0]:
                capacity.append(hindsight(cell, cut))
    roster = {b["rec"]: b["source"] for b in blocks}
    readout = job.grouped_readout(rows, roster, job.SEEDS)
    matched = [readout["risk"][k]["switches"] for k in ARMS]
    if len(fit_rows) != 27 or len(rows) != 72 or matched != [95769]*3:
        raise ValueError("Unexpected matched readout")
    bound = job.grouped_readout(capacity, roster, [job.SEEDS[0]])
    run.once(OUT / "analysis.json", run.stable(dict(registration=binding, fits=54, calibrated=fit_rows,
        summary=readout, hindsight=bound, hindsight_never_used_for_selection=True,
        hindsight_records=[dict(recording=r["recording"], source=r["source"], **r["metrics"]["oracle"]["risk"])
                           for r in capacity],
        calibration_access=False, confirmation_access=False, deployment=False)))


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

        def beat(**d):
            d.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", d)
            if d.get("step") == timing.STEPS:
                print(d, flush=True)

        execute(key, identity, cut, freeze, beat, args.replay)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
