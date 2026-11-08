"""Fit and score a variable-count query allocation without opening confirmation."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_matched_direction as direction
from src.world_model import m3w_optional_allocation as model, m3w_matched_support_moments as moments

job, run, np, torch = direction.job, direction.run, direction.np, direction.torch
OUT = job.OUT.parent / "optional_allocation_v2"
PRIVATE = job.PRIVATE.parent / OUT.name
FACTORS = (0., .002, .01, .04)
ARMS = ("joint", "gain", "priced")
ALL = tuple(f"{k}_{i}" for i in range(len(FACTORS)) for k in ARMS) + ("forced", "calibrated")
PAIRS = {f"{a}_vs_{'cv' if b == job.CV else b}": (b, a, job.CV)
         for a, b in ((job.OLS, job.CV), *((a, b) for a in ALL for b in (job.CV, job.OLS)),
                      *((f"joint_{i}", f"{k}_{i}") for k in ARMS[1:] for i in range(len(FACTORS))))}


def register():
    parent, identity, cut, freeze = direction.register()
    prior = direction.OUT / "recipient_capacity_replay.json"
    if run.read(prior)["circle_summary"]["risk_infeasible"] != 12:
        raise ValueError("Require the verified fixed-count capacity result")
    file = OUT / "registration.json"
    run.once(file, run.stable(dict(parent=parent, direction_capacity=run.artifact(prior),
        original_2pct=True, code=run.artifact(Path(__file__)), solver=run.artifact(Path(model.__file__)),
        model=run.artifact(Path(moments.__file__)), seeds=job.SEEDS, fit_steps=500, checkpoint_every=100,
        threads=4, workers=0, quarter_bank=True, counts_are_maxima=True, extra_excess_fraction=FACTORS,
        capacity="minimum of two and inherited past-only query count",
        primary="strict_joint_vs_count_matched_priced", original_scoring_risk_unchanged=True,
        calibration_access=False, confirmation_access=False)))
    return run.sha(file), identity, cut, freeze


def prepare(identity, beat):
    old = job.prior.prior.old
    blocks, x, radii, truth, cursor = [], [], [], [], 0
    for rec, g, h, cv, ols, bank, y, valid, inverse, counts, w in old.blocks(identity, beat):
        bank = cv[:, None]+.25*(bank-cv[:, None])
        features, scale = old.features(h, bank, cv, inverse)
        rad = np.linalg.norm(bank-cv[:, None], axis=-1).max(2)
        raw = job.errors(cv, y, valid)[0]
        known = valid.any(1)
        err = np.stack([job.errors(p, y, valid)[0] for p in bank.swapaxes(0, 1)], 1)
        gain = np.where(known[:, None], (raw[:, None]-err)/scale[:, None], 0.)
        if np.any(abs(gain) > rad/scale[:, None]+1e-8):
            raise ValueError("Observed ADE gain exceeds path disagreement envelope")
        ols_change = (np.linalg.norm(ols-cv, axis=-1).max(1)[:, None]
                      - np.linalg.norm(ols[:, None]-bank, axis=-1).max(2))
        s = slice(cursor, cursor+len(cv))
        blocks.append(dict(rec=rec, source=g, cv=cv, ols=ols, bank=bank, target=y, valid=valid,
                           inverse=inverse, counts=np.minimum(counts, 2), weight=w, sl=s, scale=scale,
                           radius=rad, ols_change=ols_change))
        x.append(features.astype("float32"))
        radii.append((rad/scale[:, None]).astype("float32"))
        truth.append((np.where(known, raw/scale, 0.).astype("float32"), gain.astype("float32"), known, raw))
        cursor += len(cv)
    if len(blocks) != 24 or cursor != 69746:
        raise ValueError("The fixed original fitting population changed")
    return (torch.from_numpy(np.concatenate(x)), torch.from_numpy(np.concatenate(radii)),
            list(map(np.concatenate, zip(*truth, strict=True))), blocks)


def fit(binding, fold, x, radius, truth, groups, beat, replay):
    held, seed = fold["held"], fold["seed"]
    use = [groups[g] for g in fold["fit"]]
    raw = truth[-1]
    vals = raw[np.concatenate(use)]
    easy_cut = float(np.quantile(vals[np.isfinite(vals) & (vals > 0.)], .25))
    ref, gain, known = map(torch.from_numpy, truth[:3])
    easy = torch.from_numpy(known.numpy() & (raw > 0.) & (raw <= easy_cut))
    key = dict(registration=binding, held=held, seed=seed, fit=fold["fit"],
               calibration=fold["calibration"], easy_cut=easy_cut)
    p = PRIVATE / f"fit_{held}_{seed}.pt"
    torch.manual_seed(seed)
    net = moments.SupportMoments(x.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed+7919)
    step, trace = 0, []
    if p.exists():
        state = torch.load(p, weights_only=True, map_location="cpu")
        if state["key"] != key or not 0 < state["step"] <= 500:
            raise ValueError("Checkpoint identity mismatch")
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        step, trace = state["step"], state["trace"]
    if replay and step != 500:
        raise ValueError("Replay may not refit")
    for step in range(step+1, 501):
        ix = direction.timing.draw_batch(use, 128, rng)
        loss = moments.objective(net(x[ix], radius[ix]), ref[ix], gain[ix], known[ix], easy[ix], radius[ix], True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(key=key, model=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.get_state(), step=step, trace=trace), tmp)
            tmp.replace(p)
            beat(stage="checkpoint", held=held, seed=seed, step=step, loss=trace[-1])
    return net.eval(), run.artifact(p)


@torch.inference_mode()
def infer(net, x, radius, b):
    ref, cost, logits = net(x[b["sl"]], radius[b["sl"]])[:3]
    p_unknown = (1.-logits[:, 0].sigmoid()).numpy().astype(float)[:, None]
    scale = b["scale"]
    ref = ref.numpy().astype(float)*scale[:, None]
    cost = cost.numpy().astype(float)*scale[:, None, None]
    cv = cost[:, :, 0]
    ols = cv + p_unknown*(b["radius"]+b["ols_change"])
    excess = cost[:, :, 1:]-.02*ref[:, None, :]
    legal = b["radius"] > 0.
    value = np.where(legal, np.minimum(cv, ols), -np.inf)
    best = value.argmax(1)
    on = job.model.pick(value[np.arange(len(best)), best], legal.any(1),
                        b["inverse"], b["counts"], -np.inf)
    return (np.stack([model.allocate(cv, ols, excess, legal, b["inverse"], b["counts"], b["radius"], factor)
                      for factor in FACTORS]), np.where(on, best, -1))


def paths(b, choice):
    p = b["cv"].copy()
    ids = np.flatnonzero(choice >= 0)
    p[ids] = b["bank"][ids, choice[ids]]
    return p


def calibrate(cells, cut):
    trials = []
    for j, factor in enumerate(FACTORS):
        by_group, passes, switches = {}, 0, 0
        for b, actions in cells:
            p = paths(b, actions[j, 0])
            m, offset, increment, denom = job.calibration_cell(b, p, cut)
            take = actions[j, 0] >= 0
            w = b["weight"][take]
            switches += int(take.sum())
            passes += bool(m["ref_easy"][take].sum() > 0. and m["all_excess"][take].sum() <= 0.
                           and m["easy_excess"][take].sum() <= 0.
                           and m["easy_signed"][take].sum() <= .02*m["ref_easy"].sum())
            by_group.setdefault(b["source"], []).append([w@m["utility"][take], offset+w@increment[take], denom])
        both = [np.sum(a, axis=0) for a in by_group.values()]
        g = np.mean([100*c[:2]/c[2] for c in both], axis=0)
        trials.append(dict(factor=factor, gain_both=g.tolist(), passes=passes, views=len(cells), switches=switches))
    choice = job.model.select([dict(r, threshold=i) for i, r in enumerate(trials)], tuple(range(len(trials))))
    return choice, trials


def execute(key, identity, cut, freeze, beat, replay):
    x, radius, truth, blocks = prepare(identity, beat)
    groups = job.prior.prior.group_ids(blocks)
    rows, fits, changed = [], [], np.zeros((len(FACTORS), 2), dtype=np.int64)
    for fold in freeze["calibration"]:
        held, seed = fold["held"], fold["seed"]
        if job.model.roles(groups, held) != (fold["fit"], fold["calibration"]):
            raise ValueError("Fold role mismatch")
        net, artifact = fit(key, fold, x, radius, truth, groups, beat, replay)
        output = []
        for b in blocks:
            if b["source"] in (*fold["calibration"], held):
                output.append((b, infer(net, x, radius, b)))
        selected, ladder = calibrate([(b, v[0]) for b, v in output if b["source"] != held], cut)
        receipt = run.stable(dict(key=key, checkpoint=artifact, held=held, seed=seed,
                                  fit=fold["fit"], calibration=fold["calibration"], selected=selected, ladder=ladder))
        file = PRIVATE / f"policy_{held}_{seed}.json"
        if replay and not file.exists():
            raise ValueError("Replay may not choose a policy")
        run.once(file, receipt)
        fits.append(receipt)
        for b, (actions, forced) in output:
            if b["source"] != held:
                continue
            pp = {job.CV: b["cv"], job.OLS: b["ols"], "forced": paths(b, forced)}
            for j, a in enumerate(actions):
                counts = [np.bincount(b["inverse"][r >= 0], minlength=len(b["counts"])) for r in a]
                if np.any(counts[0] > b["counts"]):
                    raise ValueError("Exceeded causal query maximum")
                np.testing.assert_array_equal(counts[0], counts[1])
                np.testing.assert_array_equal(counts[0], counts[2])
                for k, r in zip(ARMS, a, strict=True):
                    pp[f"{k}_{j}"] = paths(b, r)
                changed[j] += np.sum(a[0] != a[1:], axis=1)
            pp["calibrated"] = b["cv"] if selected is None else pp[f"joint_{selected}"]
            row = direction.SCORE(pp, PAIRS, b["target"], b["valid"], b["inverse"], b["rec"],
                                  easy_cut=cut["easy"], hard_cut=cut["hard"])
            rows.append(dict(row, seed=seed, source=held))
    if len(fits) != 27 or len(rows) != 72:
        raise ValueError("Require complete source and seed roster")
    summary = job.grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, job.SEEDS)
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, fits=fits, summary=summary,
        changed_rows_vs_same_count_controls=changed.tolist(), no_official_calibration=True,
        confirmation_access=False, deployment=False)))


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
            if v.get("step") == 500:
                print(v, flush=True)

        execute(key, identity, cut, freeze, beat, args.replay)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
