"""Run paired past-sampling diagnostics without changing labels or risk floor."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_shared_correction_audit as previous
from src.world_model import m3w_past_sampling as method


prior, run, cal, np = previous.prior, previous.run, previous.cal, previous.np
OUT = previous.OUT.parent/"past_sampling_audit_v1"
PRIVATE = previous.PRIVATE.parent/OUT.name
PAIRS = {f"{a}_vs_{'cv' if b == cal.BASE else b}": (b, a, cal.BASE)
         for a, b in (
             *((a, cal.BASE) for a in (*method.SCHEDULES, cal.OLS)),
             *((a, "recent") for a in method.SCHEDULES if a != "recent"),
             ("wide_uniform", "wide_endpoints"), ("wide_dense", "wide_uniform"),
             ("wide_uniform", cal.OLS),
         )}


def register():
    binding, identity, cut = previous.register()
    path = OUT/"registration.json"
    run.once(path, run.stable(dict(code=run.artifact(Path(__file__)), method=run.artifact(Path(method.__file__)),
        parent_shared_registration=binding, parent_choices=run.artifact(prior.OUT/"freeze.json"),
        schedules=method.SCHEDULES, primary="wide_uniform_vs_recent_paired_conservative_both_cohorts",
        secondary="same_span_four_point_layout_and_seven_point_density_controls",
        same_anchors=True, same_future_labels=True, same_frozen_recipients=True, same_native_cv_floor=True,
        fit_or_threshold_selection=False, confirmation_access=False, risk_rule_changed=False)))
    return run.sha(path), identity, cut


def infer_paths(history, f, action):
    paths = {cal.BASE: f["cv"], cal.OLS: f["ols"]}
    paths.update(method.candidates(history, f["cv"], f["ols"], action > 0))
    return paths


def inference(binding, clip, choice, manifest, inventory, meta):
    f, action = previous.arrays(clip, choice)
    rec = clip["sequence"]
    reader = cal.reader(manifest, rec, meta, inventory["provenance"])
    paths = infer_paths(prior.base.history(reader, f), f, action)
    digest = prior.base.train_source.prep.array_hash
    mark = dict(registration=binding, recording=rec, choices=choice["choices"],
                parent=clip["artifact"], paths={k: digest(v) for k, v in paths.items()},
                eligible_recipients=int((action > 0).sum()))
    return mark, (reader, f, paths)


def freeze(binding, beat):
    manifest, parent, _, inv, meta = prior.target_context()
    choices = run.read(prior.OUT/"freeze.json")["clips"]
    marks = []
    for i, (clip, choice) in enumerate(zip(parent["clips"], choices, strict=True), 1):
        p = PRIVATE/f"prefix_{clip['sequence']}.json"
        if not p.exists():
            mark, _ = inference(binding, clip, choice, manifest, inv, meta)
            run.once(p, run.stable(mark))
        mark = run.read(p)
        if mark["registration"] != binding or mark["choices"] != choice["choices"] or mark["parent"] != clip["artifact"]:
            raise ValueError("Frozen path alignment changed")
        marks.append(mark)
        if i % 5 == 0:
            beat(stage="freeze", finished=i)
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, labels_read=False, recordings=marks)))


def score_block(rec, group, f, paths, target, valid, cut):
    q = np.repeat(f["frames"], np.diff(f["offsets"]))
    row = prior.base.score_recording(paths, PAIRS, target, valid, q, rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
    return dict(row, seed=0, source=group)


def fitting(binding, identity, cut, beat, replay):
    p = PRIVATE/"fitting.json"
    if not p.exists():
        if replay:
            raise ValueError("Replay may not open fitting labels")
        train = prior.base.train_source
        data = train.fitting_only(identity, beat)
        prep = train.prep
        cfg = run.read(train.source.CONFIG)
        admission = prep.load_admission(ROOT, ROOT/cfg["admission"])
        inventory = run.read(train.bank.old.HOME/"movingdronecrowd_intake_v1/analysis.json")
        meta = {r["clip"]: r for r in inventory["recordings"]}
        lo, rows = 0, []
        for rec in data["recordings"]:
            cell = admission["recordings"][rec]
            clip = meta[cell["clip"]]
            file = ROOT/cell["source_annotation_path"]
            prep.require_fit_record(admission, rec, "fit", file)
            raw = prep.mot_record(file.read_bytes(), cell["clip"], range(1, clip["image_frames"]+1))
            times = np.arange(7, raw.valid.shape[1]-12)
            counts = np.array([raw.valid[:, t-7:t+1].all(1).sum() for t in times])
            n = clip["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
            hi = lo+n
            g = cell["exclusion_group"]
            if counts.sum() != n or not np.all(data["sites"][lo:hi] == g):
                raise ValueError("Fit query membership mismatch")
            h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
            cv = data["geometry"][lo:hi, 332:356].reshape(-1, 12, 2).astype(float)
            ols = prior.base.motion.slope(h[:, -4:])[:, None]*previous.LEAD
            f = dict(cv=cv, ols=ols, frames=times[counts > 0], offsets=np.r_[0, np.cumsum(counts[counts > 0])])
            a = prior.method.menu(prior.method.backtests(h), np.any(cv != ols, axis=(1, 2)))["three_origins_2"]
            paths = infer_paths(h, f, a)
            rows.append(score_block(rec, g, f, paths, data["target"][lo:hi], data["valid"][lo:hi], cut))
            lo = hi
        if lo != len(data["ids"]):
            raise ValueError("Fitting population changed")
        run.once(p, run.stable(dict(registration=binding, rows=rows)))
    fit = run.read(p)
    if fit["registration"] != binding:
        raise ValueError("Fit score cache mismatch")
    return fit["rows"]


def aggregate(rows):
    return prior.base.grouped_readout(rows, {r["recording"]: r["source"] for r in rows}, (0,))


def score(binding, identity, cut, beat, replay):
    manifest, parent, roster, inv, meta = prior.target_context()
    freeze = run.read(OUT/"freeze.json")
    if freeze["registration"] != binding or freeze["labels_read"]:
        raise ValueError("Require all paths frozen before labels")
    choices = run.read(prior.OUT/"freeze.json")["clips"]
    rows = []
    for i, (clip, choice, expected) in enumerate(zip(parent["clips"], choices, freeze["recordings"], strict=True), 1):
        rec = clip["sequence"]
        p = PRIVATE/f"score_{rec}.json"
        if not p.exists():
            if replay:
                raise ValueError("Replay may not open target labels")
            mark, (reader, f, paths) = inference(binding, clip, choice, manifest, inv, meta)
            if mark != expected:
                raise ValueError("Path hashes changed")
            target, valid = cal.label_arrays(reader, f)
            r = score_block(rec, roster[rec], f, paths, target, valid, cut)
            run.once(p, run.stable(dict(receipt=mark, row=r)))
        r = run.read(p)
        if r["receipt"] != expected:
            raise ValueError("Stale target score")
        rows.append(r["row"])
        if i % 5 == 0:
            beat(stage="score", finished=i)
    source, target = aggregate(fitting(binding, identity, cut, beat, replay)), aggregate(rows)
    old = run.read(previous.OUT/"analysis.json")
    for key, got in (("fitting", source), ("opened_calibration", target)):
        ref = old[key]["summary"]
        if (got["effects"]["recent_vs_cv"] != ref["effects"][previous.RAW+"_vs_cv"]
                or got["risk"]["recent"] != ref["risk"][previous.RAW]
                or got["effects"][cal.OLS+"_vs_cv"] != ref["effects"][cal.OLS+"_vs_cv"]):
            raise ValueError("Original controls did not reproduce")
        if any(got["risk"][a]["switches"] > got["risk"]["recent"]["switches"] for a in method.SCHEDULES):
            raise ValueError("New recipient outside frozen intervention set")
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, fitting=source, opened_calibration=target,
        fitted_models=0, primary="wide_uniform_vs_recent", confirmation_opened=False, risk_rule_changed=False,
        no_threshold_search=True, deployment=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "freeze", "score", "replay"))
    a = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut = register()

        def beat(**v):
            v.update(pid=os.getpid(), phase=a.phase)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            print(v, flush=True)

        if a.phase == "freeze":
            freeze(binding, beat)
        elif a.phase in ("score", "replay"):
            score(binding, identity, cut, beat, a.phase == "replay")


if __name__ == "__main__":
    main()
