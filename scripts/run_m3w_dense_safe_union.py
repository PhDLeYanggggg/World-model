"""Repair the dense search using per-action predicted feasibility certificates."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dense_motion_calibration as base


run, np, torch = base.run, base.np, base.torch
OUT = base.PUBLIC.parent/"dense_safe_union_v1"
PRIVATE = base.PRIVATE.parent/OUT.name
PAIRS = {f"{a}_vs_{'cv' if b == base.cal.BASE else b}": (b,a,base.cal.BASE) for a,b in (
    (base.cal.OLS, base.cal.BASE), ("union", base.cal.BASE), ("matched", base.cal.BASE),
    ("union", base.cal.OLS), ("union", "matched"), ("union", "independent"),
)}


def choose_union(ref, moments, legal, budget=.02):
    gain, harm, easy = moments.transpose(2, 0, 1)
    safe = (legal[:, None] & (gain > 0.) & (ref > 0.).all(1)[:, None] &
            (harm <= budget*ref[:, :1]) & (easy <= budget*ref[:, 1:]))
    score = np.where(safe, gain, -np.inf)
    k = score.argmax(1)
    on = safe[np.arange(len(k)), k]
    action = np.where(on, k+1, 0).astype(np.uint8)
    if not on.any():
        return action, None
    r = ref[on].sum(0)
    ratios = np.array([harm[on, k[on]].sum(), easy[on, k[on]].sum()])/r
    if (ratios > budget+1e-9).any():
        raise ValueError("Per-action certificates must survive pooling")
    return action, ratios.tolist()


def register():
    _, _, manifest, parent, roster, cut, original = base.register()
    freeze = run.read(base.PUBLIC/"forecast_freeze.json")
    audit = run.read(base.PUBLIC/"predicted_feasibility.json")
    if (freeze["calibration_labels_used"] or audit["future_labels_read"]
            or audit["joint_empty_views"] != 120 or audit["positive_gain_singleton_views"] != 120):
        raise ValueError("Require the observed label-free price-ladder failure and repair opportunity")
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(
        method=run.artifact(Path(__file__)),
        moment_freeze=run.artifact(base.PUBLIC/"forecast_freeze.json"),
        diagnostic=run.artifact(base.PUBLIC/"predicted_feasibility.json"),
        selection="highest positive gain among individually feasible actions",
        budget=.02, matched="same per-query amplitude counts assigned by causal hindcast",
        parents_adjusted_using_outcomes=False, no_fits=True, confirmation_access=False,
        scope="label_free_repair_of_opened_calibration_diagnostic",
    )))
    return run.sha(reg), manifest, parent, roster, cut, original, freeze


@torch.inference_mode()
def infer(binding, manifest, parent, roster, _, original, freeze, beat):
    nets = base.network_receipt(freeze["registration"])
    inventory = run.read(base.cal.CACHE)
    meta = {r["sequence_id"]:r for r in inventory["recordings"] if r["sequence_id"] in roster}
    clips = []
    for i, (clip, parent_sel) in enumerate(zip(parent["clips"], freeze["clips"], strict=True), 1):
        rec = clip["sequence"]
        mark = PRIVATE/f"choice_{rec}.json"
        if not mark.exists():
            src = base.cal.reader(manifest, rec, meta, inventory["provenance"])
            f = base.frozen_paths(clip)
            x, norm, b, hind = base.motion.features(base.history(src, f))
            x, b = torch.from_numpy(x), torch.from_numpy(b.astype(np.float32))
            legal = np.any(f["cv"] != f["ols"], axis=(1, 2))
            selections, diag = [], []
            for net in nets:
                pred = [net(t, v) for t,v in zip(x.split(8192), b.split(8192), strict=True)]
                ref, cost = [torch.cat(z).numpy() for z in zip(*pred, strict=True)]
                action, ratios = choose_union(ref*norm[:, None], cost*norm[:, None, None], legal)
                matched = base.motion.matched_hindcast(action, hind, f["offsets"], legal)
                selections.append(np.stack((action, matched)))
                diag.append(dict(predicted_risks=ratios, switches=int((action > 0).sum())))
            p = mark.with_suffix(".npz")
            tmp = p.with_suffix(".pending.npz")
            np.savez_compressed(tmp, choices=np.stack(selections))
            tmp.replace(p)
            run.once(mark, run.stable(dict(registration=binding, recording=rec, choices=run.artifact(p),
                                          prior_choices=parent_sel["choices"], diagnostics=diag)))
        r = run.read(mark)
        if r["registration"] != binding or r["prior_choices"] != parent_sel["choices"]:
            raise ValueError("Stale union inference cache")
        clips.append(r)
        if i % 5 == 0:
            beat(stage="infer", finished=i, total=len(roster))
    run.once(OUT/"forecast_freeze.json", run.stable(dict(
        registration=binding, clips=clips, calibration_labels_used=False)))


def score(binding, manifest, parent, roster, cut, original, freeze, beat, replay):
    selections = run.read(OUT/"forecast_freeze.json")
    if (selections["registration"] != binding or selections["calibration_labels_used"]
            or len(selections["clips"]) != len(roster)):
        raise ValueError("Complete union predictions must be frozen before scoring")
    inventory = run.read(base.cal.CACHE)
    meta = {r["sequence_id"]:r for r in inventory["recordings"] if r["sequence_id"] in roster}
    rows = []
    for i,(clip, a, b) in enumerate(zip(parent["clips"], selections["clips"], freeze["clips"], strict=True), 1):
        rec = clip["sequence"]
        if a["recording"] != rec or b["sequence"] != rec or a["prior_choices"] != b["choices"]:
            raise ValueError("Both choices must match the original recording")
        cache = PRIVATE/f"score_{rec}.json"
        if not cache.exists():
            if replay:
                raise ValueError("Replay does not reopen labels")
            src = base.cal.reader(manifest, rec, meta, inventory["provenance"])
            f = base.frozen_paths(clip)
            y, valid = base.cal.label_arrays(src, f)
            q = np.repeat(f["frames"], np.diff(f["offsets"]))
            with np.load(run.checked(a["choices"]), allow_pickle=False) as z:
                new = z["choices"]
            with np.load(run.checked(b["choices"]), allow_pickle=False) as z:
                old = z["choices"][:,0]
            result = []
            for seed, acts, p in zip(base.cal.SEEDS, new, old, strict=True):
                paths = {base.cal.BASE:f["cv"], base.cal.OLS:f["ols"]}
                paths.update({k:base.motion.make_path(f["cv"],f["ols"],action)
                              for k,action in zip(("union","matched","independent"), (*acts,p), strict=True)})
                r = base.score_recording(paths, PAIRS, y, valid, q, rec,
                                         easy_cut=cut["easy"], hard_cut=cut["hard"])
                result.append(dict(r, source=roster[rec], seed=seed))
            run.once(cache, run.stable(dict(registration=binding, choices=a["choices"], rows=result)))
        r = run.read(cache)
        if r["registration"] != binding or r["choices"] != a["choices"]:
            raise ValueError("Scoring cache mismatch")
        rows.extend(r["rows"])
        if i % 5 == 0:
            beat(stage="score", finished=i, total=len(roster))
    s = base.grouped_readout(rows, roster, base.cal.SEEDS)
    if (s["risk"]["union"]["switches"] != s["risk"]["matched"]["switches"]
            or s["effects"][f"{base.cal.OLS}_vs_cv"] != original["summary"]["effects"][f"{base.cal.OLS}_vs_cv"]):
        raise ValueError("Original reference or matched intervention count changed")
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, summary=s, no_new_fits=True,
        selected_using_outcomes=False, calibration_diagnostic_only=True, risk_rule_changed=False,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register","infer","score","replay"))
    a = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = register()

        def beat(**v):
            v.update(pid=os.getpid(), phase=a.phase)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            print(v, flush=True)

        if a.phase == "infer":
            infer(*args, beat)
        elif a.phase in ("score", "replay"):
            score(*args, beat, a.phase=="replay")


if __name__ == "__main__":
    main()
