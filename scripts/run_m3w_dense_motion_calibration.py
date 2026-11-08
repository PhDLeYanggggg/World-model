"""Train on original fitting sources and diagnose a broad causal motion policy."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 .venv-pytorch")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_calibration as cal
from scripts import report_m3w_dronecrowd_calibration as verified
from scripts import run_m3w_easy_aligned_forecast as train_source
from src.evaluation.m3w_grouped_recording_readout import grouped_readout
from src.evaluation.m3w_recording_forecast_comparison import score_recording
from src.world_model import m3w_dense_motion_moments as motion
from src.world_model import m3w_native_forecast as fit


run, np, torch = cal.run, cal.np, cal.torch
PUBLIC = cal.PUBLIC.parent/"dense_motion_calibration_v1"
PRIVATE = cal.PRIVATE.parent/PUBLIC.name
STEPS, BATCH, EVERY = 1000, 128, 100
ARM_ORDER = ("independent", "joint", "matched_hindcast")
PAIRS = {f"{a}_vs_{'cv' if b==cal.BASE else b}": (b, a, cal.BASE) for a,b in (
    (cal.OLS, cal.BASE), ("fixed_half", cal.BASE), ("joint", cal.BASE),
    ("joint", cal.OLS), ("joint", "fixed_half"), ("joint", "independent"),
    ("joint", "matched_hindcast"), ("independent", cal.BASE),
    ("independent", cal.OLS), ("matched_hindcast", cal.BASE),
)}


def register():
    official, parent = verified.verify_artifacts(ROOT, cal.PUBLIC, cal.roles.DEST)
    if verified.summarize(official)["selected_arm"] is not None:
        raise ValueError("This experiment follows the original failed calibration")
    jobs = train_source.registration()
    population = train_source.PUBLIC/"fit_population.json"
    fit_data = run.read(population)
    if fit_data["rows"] != 69746 or len(fit_data["groups"]) != 9 or fit_data["A_target_opened"]:
        raise ValueError("Original fitting split required")
    manifest = run.read(cal.roles.DEST)
    parent_freeze = run.read(cal.PUBLIC/"forecast_freeze.json")
    roster = {r:g for r,g in run.read(cal.roles.GROUPS)["recording_group"].items()
              if manifest["recording_role_index"][r] == "calibration"}
    if len(roster) != 40 or len(set(roster.values())) != 15 or set(roster) != {c["sequence"] for c in parent_freeze["clips"]}:
        raise ValueError("Only the complete calibration roster can be evaluated")
    cut = run.read(cal.REG)["cut"]
    registration = PUBLIC/"registration.json"
    run.once(registration, run.stable(dict(
        parent=parent, code=run.artifact(Path(__file__)), method=run.artifact(Path(motion.__file__)),
        fit_population=run.artifact(population), fit_ids=jobs[0]["identity"]["train_ids_sha256"],
        seeds=cal.SEEDS, fit_steps=STEPS, fit_batch=BATCH, lr=.001, workers=0, torch_threads=4,
        level=motion.LEVELS.tolist(), prices=motion.MULTIPLIERS, risk_budget=.02,
        support_failure="CV identity reported unsupported", cut=cut,
        primary=["joint_vs_history_ols4_velocity", "joint_vs_matched_hindcast"],
        role="post_calibration_diagnostic_not_new_independent_calibration",
        test_access=False, original_risk_unchanged=True,
    )))
    return run.sha(registration), jobs[0]["identity"], manifest, parent_freeze, roster, cut, official


def train(binding, identity, cut, beat):
    data = train_source.fitting_only(identity, beat)
    h = data["geometry"][:, :16].reshape(-1, 8, 2)
    cv = data["geometry"][:, 332:356].reshape(-1, 12, 2).astype(float)
    lead = np.arange(1., 13.)[None, :, None]
    ols = motion.slope(h[:, -4:])[:, None]*lead
    x, norm, bound, _ = motion.features(h)
    ref, target = motion.targets(cv, ols, data["target"], data["valid"], norm, cut["easy"])
    x, bound, ref, target = [torch.from_numpy(z.astype(np.float32)) for z in (x, bound, ref, target)]
    receipts = []
    for seed in cal.SEEDS:
        torch.manual_seed(seed)
        net = motion.DenseMoments()
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        rng = torch.Generator().manual_seed(seed+7919)
        path = PRIVATE/f"seed_{seed}.pt"
        key = dict(registration=binding, seed=seed)
        step, trace = 0, []
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["identity"] != key or not 0 < saved["step"] <= STEPS:
                raise ValueError("Invalid resume checkpoint")
            net.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            rng.set_state(saved["rng"])
            step, trace = saved["step"], saved["trace"]
        for i in range(step, STEPS):
            ix = fit.draw_batch(data["groups"], BATCH, rng)
            r, m = net(x[ix], bound[ix])
            loss = (torch.nn.functional.mse_loss(r, ref[ix]) +
                    torch.nn.functional.mse_loss(m, target[ix]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite moment fit")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            if (i+1) % EVERY == 0:
                trace.append(float(loss.detach()))
                tmp = path.with_suffix(".pending")
                torch.save(dict(identity=key, step=i+1, model=net.state_dict(),
                                optimizer=opt.state_dict(), rng=rng.get_state(), trace=trace), tmp)
                tmp.replace(path)
                beat(stage="fit_checkpoint", seed=seed, step=i+1, loss=trace[-1])
        receipts.append(dict(seed=seed, checkpoint=run.artifact(path), step=STEPS, loss_trace=trace))
    run.once(PRIVATE/"training.json", run.stable(dict(registration=binding, fits=receipts, rows=len(h), sources=9)))


def network_receipt(binding):
    receipt = run.read(PRIVATE/"training.json")
    if receipt["registration"] != binding or tuple(f["seed"] for f in receipt["fits"]) != cal.SEEDS:
        raise ValueError("All trained seeds required")
    nets = []
    for f in receipt["fits"]:
        state = torch.load(run.checked(f["checkpoint"]), map_location="cpu", weights_only=True)
        if state["step"] != STEPS or state["identity"] != dict(registration=binding, seed=f["seed"]):
            raise ValueError("Trained checkpoint mismatch")
        net = motion.DenseMoments().eval()
        net.load_state_dict(state["model"])
        nets.append(net)
    return nets


def frozen_paths(cell):
    with np.load(run.checked(cell["artifact"]), allow_pickle=False) as z:
        return {k:z[k] for k in ("cv", "ols", "frames", "offsets", "agents", "origin")}


def history(source, f):
    blocks = []
    np.testing.assert_array_equal(source.query_frames, f["frames"])
    for frame, lo, hi in zip(f["frames"], f["offsets"][:-1], f["offsets"][1:], strict=True):
        raw = source.get_inputs(int(frame))
        take = raw["history_complete"]
        h = raw["positions"][take]
        np.testing.assert_array_equal(raw["agent_ids"][take], f["agents"][lo:hi])
        np.testing.assert_array_equal(h[:, -1], f["origin"][lo:hi])
        blocks.append(h)
    return np.concatenate(blocks)


@torch.inference_mode()
def infer_record(source, f, nets):
    x, norm, bound, hind = motion.features(history(source, f))
    x, b = torch.from_numpy(x), torch.from_numpy(bound.astype(np.float32))
    choices, meta = [], []
    legal = np.any(f["cv"] != f["ols"], axis=(1, 2))
    for net in nets:
        pred = [net(a, b0) for a,b0 in zip(x.split(8192), b.split(8192), strict=True)]
        ref, cost = [torch.cat(z).numpy() for z in zip(*pred, strict=True)]
        independent, joint, price = motion.choose(ref*norm[:, None], cost*norm[:, None, None], f["offsets"], legal)
        match = motion.matched_hindcast(joint, hind, f["offsets"], legal)
        choices.append(np.stack((independent, joint, match)))
        meta.append(dict(price=price, independent_switches=int((independent > 0).sum()),
                         joint_switches=int((joint > 0).sum())))
    return np.stack(choices), meta


def infer(binding, manifest, parent, roster, beat):
    nets = network_receipt(binding)
    inventory = run.read(cal.CACHE)
    meta = {r["sequence_id"]:r for r in inventory["recordings"] if r["sequence_id"] in roster}
    clips = []
    for i,c in enumerate(parent["clips"], 1):
        rec = c["sequence"]
        p = PRIVATE/f"choice_{rec}.npz"
        mark = p.with_suffix(".json")
        if not mark.exists():
            source = cal.reader(manifest, rec, meta, inventory["provenance"])
            f = frozen_paths(c)
            choices, diag = infer_record(source, f, nets)
            tmp = p.with_suffix(".pending.npz")
            np.savez_compressed(tmp, choices=choices)
            tmp.replace(p)
            run.once(mark, run.stable(dict(registration=binding, parent=c["artifact"],
                                          sequence=rec, choices=run.artifact(p), diagnostics=diag)))
        r = run.read(mark)
        if r["registration"] != binding or r["parent"] != c["artifact"]:
            raise ValueError("Stale forecast")
        run.checked(r["choices"])
        clips.append(r)
        beat(stage="choices_frozen", sequence=rec, finished=i, total=len(roster))
    run.once(PUBLIC/"forecast_freeze.json", run.stable(dict(
        registration=binding, clips=clips, calibration_labels_used=False,
        training=run.artifact(PRIVATE/"training.json"))))


def score(binding, manifest, parent, roster, cut, original, beat, replay):
    frozen = run.read(PUBLIC/"forecast_freeze.json")
    if frozen["registration"] != binding or frozen["calibration_labels_used"] or len(frozen["clips"]) != 40:
        raise ValueError("Require the complete frozen policy before labels")
    inventory = run.read(cal.CACHE)
    meta = {r["sequence_id"]:r for r in inventory["recordings"] if r["sequence_id"] in roster}
    rows = []
    for i,(clip, sel) in enumerate(zip(parent["clips"], frozen["clips"], strict=True), 1):
        rec = clip["sequence"]
        if sel["sequence"] != rec or sel["parent"] != clip["artifact"]:
            raise ValueError("Frozen choices have changed roster")
        path = PRIVATE/f"score_{rec}.json"
        if not path.exists():
            if replay:
                raise ValueError("Replay does not reopen outcomes")
            source = cal.reader(manifest, rec, meta, inventory["provenance"])
            f = frozen_paths(clip)
            y, valid = cal.label_arrays(source, f)
            with np.load(run.checked(sel["choices"]), allow_pickle=False) as z:
                choices = z["choices"]
            queries = np.repeat(f["frames"], np.diff(f["offsets"]))
            result = []
            for seed, actions in zip(cal.SEEDS, choices, strict=True):
                paths = {cal.BASE:f["cv"], cal.OLS:f["ols"], "fixed_half":.5*(f["cv"]+f["ols"])}
                paths.update({k:motion.make_path(f["cv"], f["ols"], a)
                              for k,a in zip(ARM_ORDER, actions, strict=True)})
                r = score_recording(paths, PAIRS, y, valid, queries, rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
                result.append(dict(r, source=roster[rec], seed=seed))
            run.once(path, run.stable(dict(registration=binding, choices=sel["choices"], rows=result)))
        r = run.read(path)
        if r["registration"] != binding or r["choices"] != sel["choices"]:
            raise ValueError("Scoring cache mismatch")
        rows.extend(r["rows"])
        if i % 5 == 0:
            beat(stage="scored", finished=i, total=len(roster))
    summary = grouped_readout(rows, roster, cal.SEEDS)
    if summary["effects"][f"{cal.OLS}_vs_cv"] != original["summary"]["effects"][f"{cal.OLS}_vs_cv"]:
        raise ValueError("Original OLS4 accuracy readout differs")
    if summary["risk"]["joint"]["switches"] != summary["risk"]["matched_hindcast"]["switches"]:
        raise ValueError("Matched control must preserve real switches")
    run.once(PUBLIC/"analysis.json", run.stable(dict(
        registration=binding, summary=summary, original_risk_unchanged=True,
        threshold_tuned_on_calibration=False, confirmation_access=False, deployment=False,
        scientific_scope="opened_calibration_diagnostic_no_independent_confirmation",
        stage5c_executed=False, smc_enabled=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "train", "infer", "score", "replay"))
    args = cli.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, manifest, parent, roster, cut, official = register()

        def beat(**v):
            v.update(pid=os.getpid(), phase=args.phase)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            print(v, flush=True)

        if args.phase == "train":
            train(binding, identity, cut, beat)
        elif args.phase == "infer":
            infer(binding, manifest, parent, roster, beat)
        elif args.phase in ("score", "replay"):
            score(binding, manifest, parent, roster, cut, official, beat, args.phase=="replay")


if __name__ == "__main__":
    main()
