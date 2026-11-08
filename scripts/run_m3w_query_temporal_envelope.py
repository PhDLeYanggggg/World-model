"""Train query-local temporal envelopes with matched recipients and radii."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_nested_recipient_calibration as job
from src.world_model import m3w_query_temporal_envelope as model
from src.world_model.m3w_native_forecast import draw_batch

run, torch, np = job.run, job.torch, job.np
OUT = job.OUT.parent / "query_temporal_envelope_v1"
PRIVATE = job.PRIVATE.parent / OUT.name
STEPS, EVERY = 500, 100
SHAPES = ("flat", "early", "late", "learned")
ARMS = (*SHAPES, "cal_static", "cal_learned")
PAIRS = {f"{a}_vs_{'cv' if b == job.CV else b}": (b, a, job.CV)
         for a, b in ((job.OLS, job.CV), *((a, b) for a in ARMS for b in (job.CV, job.OLS)),
                      ("learned", "flat"), ("learned", "early"), ("learned", "late"))}


def register():
    parent, identity, cut = job.register()
    p = job.OUT / "freeze.json"
    freeze = run.read(p)
    if (freeze["registration"] != parent or not freeze["all_outer_choices_frozen_before_score"]
            or freeze["confirmation_access"] or freeze["official_calibration_access"]):
        raise ValueError("Require the unchanged development-only nomination run")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(parent=run.artifact(p), parent_registration=parent,
        code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)), steps=STEPS,
        every=EVERY, seeds=job.SEEDS, cpu_threads=4, workers=0, radius_fraction=.25,
        calibration_choices=list(SHAPES), primary="learned_vs_flat_paired_completion_lower",
        identical_recipients_and_per_row_radius=True, original_2pct=True,
        official_calibration_access=False, confirmation_access=False)))
    return run.sha(f), identity, cut, freeze


def inputs(x, blocks, fold, key):
    held = fold["held"]
    path = job.PRIVATE / f"nomination_{held}.json"
    nom = run.read(path)
    if (nom["registration"], nom["held"], nom["fit"], nom["cal"]) != (
            key, held, fold["fit"], fold["calibration"]):
        raise ValueError("Nominee source roles mismatch")
    sets = []
    for b, (a, legal, score) in zip(blocks, nom["nominations"], strict=True):
        a, legal, score = np.asarray(a), np.asarray(legal), np.asarray(score)
        on = job.model.pick(score, legal, b["inverse"], b["counts"], -np.inf)
        i = np.flatnonzero(on)
        d = b["bank"][i, a[i]] - b["cv"][i]
        if not np.all(np.linalg.norm(d, axis=2).max(1) > 0.):
            raise ValueError("A selected path must be nonidentity")
        scale = b["scale"][i, None, None]
        f = x[b["sl"]][i, a[i]]
        array = lambda v: torch.from_numpy(np.asarray(v, dtype=np.float32))
        sets.append((b, i, a, d, f, array(d/scale)))
    return sets, run.artifact(path)


def fit(binding, fold, sets, nominee, cut, beat, replay):
    held, seed = fold["held"], fold["seed"]
    train = [c for c in sets if c[0]["source"] in fold["fit"] and len(c[1])]
    groups, offset = {}, 0
    payload = []
    for b, i, _, _, x, displacement in train:
        valid = b["valid"][i]
        relative = np.where(valid[..., None], b["target"][i] - b["cv"][i], 0.)
        err = np.linalg.norm(relative, axis=2).sum(1) / np.maximum(valid.sum(1), 1)
        scale = b["scale"][i, None, None]
        ols = (b["ols"][i] - b["cv"][i]) / scale
        payload.append((x, displacement, torch.from_numpy((relative/scale).astype("float32")),
            torch.from_numpy(ols.astype("float32")), torch.from_numpy(valid),
            torch.from_numpy((err > 0.) & (err <= cut))))
        groups.setdefault(b["source"], []).append(np.arange(offset, offset+len(i)))
        offset += len(i)
    if set(groups) != set(fold["fit"]):
        raise ValueError("Every fitting source must contribute selected rows")
    features, displacements, truth, ols, valid, easy = map(torch.cat, zip(*payload, strict=True))
    ids = [np.concatenate(groups[g]) for g in sorted(groups)]
    path = PRIVATE / f"fit_{held}_{seed}.pt"
    key = dict(registration=binding, fit=fold["fit"], calibration=fold["calibration"], held=held,
               seed=seed, nominee=nominee, fit_easy_cut=cut)
    torch.manual_seed(seed)
    net = model.TemporalEnvelope(features.shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, trace = 0, []
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["key"] != key or not 0 < state["step"] <= STEPS:
            raise ValueError("Invalid checkpoint lineage")
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        step, trace = state["step"], state["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot fit")
    for step in range(step+1, STEPS+1):
        i = draw_batch(ids, 128, rng)
        edit, shape = net(features[i], displacements[i])
        loss = model.objective(edit, shape, truth[i], ols[i], valid[i], easy[i])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % EVERY == 0:
            trace.append(float(loss.detach()))
            temp = path.with_suffix(".pending")
            torch.save(dict(key=key, step=step, trace=trace, model=net.state_dict(),
                            optimizer=optimizer.state_dict(), rng=rng.get_state()), temp)
            temp.replace(path)
            beat(stage="checkpoint", held=held, seed=seed, step=step, loss=trace[-1])
    return net.eval(), run.artifact(path)


@torch.inference_mode()
def predict(net, cell):
    b, i, _, d, features, normalized = cell
    t = d.shape[1]
    delta = torch.from_numpy(d)
    ramp = torch.linspace(.25, 1., t, dtype=delta.dtype)
    shape = net(features, normalized)[1].to(delta.dtype)
    candidates = dict(flat=torch.ones_like(ramp), early=ramp.flip(0), late=ramp, learned=shape)
    paths = {}
    for name, v in candidates.items():
        path = b["cv"].copy()
        path[i] += model.fixed_radius(delta, v).numpy()
        np.testing.assert_allclose(np.linalg.norm(path[i] - b["cv"][i], axis=2).max(1),
                                   .25*np.linalg.norm(d, axis=2).max(1), atol=1e-9, rtol=1e-9)
        paths[name] = path
    return paths


def select(calibration, arms):
    eligible = [k for k in arms if calibration[k]["passes"] == calibration[k]["views"]
                and calibration[k]["switches"] > 0 and min(calibration[k]["gain_both"]) > 0.]
    return max(eligible, key=lambda k: min(calibration[k]["gain_both"])) if eligible else None


def run_folds(binding, cut, freeze, x, raw, blocks, beat, replay):
    rows, choices, receipts = [], [], []
    old_marks = {}
    for artifact in freeze["predictions"]:
        v = run.read(run.checked(artifact))
        old_marks[v["record"], v["seed"]] = v
    groups = job.prior.prior.group_ids(blocks)
    for fold in freeze["calibration"]:
        held, seed = fold["held"], fold["seed"]
        fit_groups, cal_groups = job.model.roles(groups, held)
        if (fit_groups, cal_groups) != (fold["fit"], fold["calibration"]):
            raise ValueError("Overlapping source roles")
        sets, nom = inputs(x, blocks, fold, freeze["registration"])
        a = raw[np.concatenate([groups[g] for g in fit_groups])]
        local_cut = float(np.quantile(a[np.isfinite(a) & (a > 0.)], .25))
        net, receipt = fit(binding, fold, sets, nom, local_cut, beat, replay)
        receipts.append(receipt)
        cal = {k: [] for k in SHAPES}
        for cell in sets:
            b, i = cell[:2]
            if b["source"] not in cal_groups:
                continue
            paths = predict(net, cell)
            for k, p in paths.items():
                cal[k].append((b, i, job.calibration_cell(b, p, cut)))
        scores = {}
        for k, v in cal.items():
            bb = [b for b, _, _ in v]
            ok = [(np.isin(np.arange(len(b["cv"])), i), c) for b, i, c in v]
            r = job.trial(bb, [np.zeros(len(b["cv"])) for b in bb], ok, 0.)
            r.pop("threshold")
            scores[k] = r
        policy = dict(static=select(scores, SHAPES[:-1]), learned=select(scores, ("learned",)))
        dest = PRIVATE / f"policy_{held}_{seed}.json"
        freeze_policy = run.stable(dict(held=held, seed=seed, fit=fit_groups, calibration=cal_groups,
            checkpoint=receipt, scores=scores, policy=policy, registration=binding))
        run.once(dest, freeze_policy)
        choices.append(freeze_policy)
        for cell in sets:
            b, i, a = cell[:3]
            if b["source"] != held:
                continue
            expected = np.full(len(a), -1)
            expected[i] = a[i]
            np.testing.assert_array_equal(expected, old_marks[b["rec"], seed]["actions"][1])
            paths = predict(net, cell)
            paths.update({job.CV: b["cv"], job.OLS: b["ols"]})
            for k, arm in policy.items():
                paths[f"cal_{k}"] = paths[arm or job.CV]
            row = job.prior.prior.prior.base.score_recording(paths, PAIRS, b["target"], b["valid"],
                b["inverse"], b["rec"], easy_cut=cut["easy"], hard_cut=cut["hard"])
            rows.append(dict(row, seed=seed, source=held))
    summary = job.grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, job.SEEDS)
    if len(receipts) != 27 or len(rows) != 72:
        raise ValueError("Incomplete experiment")
    result = run.stable(dict(registration=binding, checkpoints=receipts, calibration=choices,
        summary=summary, confirmation_access=False, official_calibration_access=False, deployment=False))
    run.once(OUT / "analysis.json", result)


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
        binding, identity, cut, freeze = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            if v.get("step") == STEPS:
                print(v, flush=True)

        x, truth, blocks = job.prior.prior.prepare(identity, beat)
        run_folds(binding, cut, freeze, x[0], truth[-1], blocks, beat, args.replay)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
