"""Diagnose shared velocity correction with frozen recipients and a norm control."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_standalone_hindcast as prior
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model import m3w_shared_correction as method


run, cal, np = prior.run, prior.cal, prior.np
OUT = prior.OUT.parent/"shared_correction_audit_v1"
PRIVATE = prior.PRIVATE.parent/OUT.name
RAW, SHARED, MAG = "frozen_hindcast", "shared_removed", "magnitude_only"
ARMS = (RAW, SHARED, MAG, cal.OLS)
PAIRS = {f"{a}_vs_{'cv' if b == cal.BASE else b}": (b, a, cal.BASE)
         for a, b in ((RAW, cal.BASE), (SHARED, cal.BASE), (MAG, cal.BASE), (cal.OLS, cal.BASE),
                      (SHARED, RAW), (MAG, RAW), (SHARED, MAG), (SHARED, cal.OLS))}
LEAD = np.arange(1., 13.)[None, :, None]


def register():
    binding, identity, cut = prior.register()
    fit = run.read(prior.OUT/"fitting.json")
    freeze = run.read(prior.OUT/"freeze.json")
    if fit["chosen"] != "three_origins_2" or fit["registration"] != binding or freeze["registration"] != binding:
        raise ValueError("Require the exact frozen deterministic half blend")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), method=run.artifact(Path(method.__file__)),
        parent_choices=run.artifact(prior.OUT/"freeze.json"), parent_fit=run.artifact(prior.OUT/"fitting.json"),
        primary="shared_removed_vs_magnitude_only_direct_conservative",
        secondary="coherence_strata_and_known_label_risk_mixture", max_norm="original per row",
        shared="leave-one-agent-out mean OLS4-minus-CV velocity among complete histories",
        thresholds_refit=False, deployment=False, confirmation_access=False, risk_rule_changed=False)))
    return run.sha(p), identity, cut


def paths(f, action):
    v = (f["ols"][:, -1]-f["cv"][:, -1])/12.
    adjusted, matched, bins = method.shared_correction(v, action > 0, f["offsets"])
    paths = {cal.BASE: f["cv"], cal.OLS: f["ols"], RAW: prior.base.motion.make_path(f["cv"], f["ols"], action),
             SHARED: f["cv"]+adjusted[:, None]*LEAD, MAG: f["cv"]+matched[:, None]*LEAD}
    return paths, bins, float(np.max(np.abs(np.linalg.norm(adjusted, axis=1)-np.linalg.norm(matched, axis=1))))


def arrays(clip, choice):
    if clip["sequence"] != choice["recording"] or clip["artifact"] != choice["parent"]:
        raise ValueError("Require frozen recording alignment")
    f = prior.base.frozen_paths(clip)
    with np.load(run.checked(choice["choices"]), allow_pickle=False) as z:
        a = z["action"]
    if not np.isin(a, [0, 2]).all():
        raise ValueError("Only half-blend or identity expected")
    return f, a


def forecast_receipt(binding, clip, choice):
    f, a = arrays(clip, choice)
    p, bins, mismatch = paths(f, a)
    hash_array = prior.base.train_source.prep.array_hash
    return dict(registration=binding, choices=choice["choices"], sequence=clip["sequence"],
                path_hashes={k: hash_array(v) for k, v in p.items()}, stratum_hash=hash_array(bins),
                matched_velocity_norm_max_diff=mismatch), (f, a, p, bins)


def freeze(binding, beat):
    _, parent, _, _, _ = prior.target_context()
    cells = run.read(prior.OUT/"freeze.json")["clips"]
    result = []
    for i, (clip, choice) in enumerate(zip(parent["clips"], cells, strict=True), 1):
        dest = PRIVATE/f"prefix_{clip['sequence']}.json"
        if not dest.exists():
            r, _ = forecast_receipt(binding, clip, choice)
            run.once(dest, run.stable(r))
        r = run.read(dest)
        if r["registration"] != binding or r["choices"] != choice["choices"]:
            raise ValueError("Prefix receipt mismatch")
        result.append(r)
        if i % 5 == 0:
            beat(stage="prefix", finished=i)
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, clips=result, labels_read=False)))


def strata(paths, selected, bins, target, valid, cut):
    base_error, _ = errors(paths[cal.BASE], target, valid)
    known = valid.any(1)
    easy = known & (base_error > 0) & (base_error <= cut)
    result = {}
    for arm in (RAW, SHARED, MAG):
        e, _ = errors(paths[arm], target, valid)
        loss = np.where(known, e-base_error, 0.)
        debit = np.linalg.norm(paths[arm]-paths[cal.BASE], axis=-1).max(1)
        cells = {}
        for j in range(-1, 3):
            on = (bins == j) & selected
            k, u, t = on & known, on & ~known, on & easy
            cells[str(j)] = dict(selected=int(on.sum()), known=int(k.sum()),
                reference=float(base_error[k].sum()), harm=float(np.maximum(loss[k], 0.).sum()),
                easy_reference=float(base_error[t].sum()), easy_harm=float(np.maximum(loss[t], 0.).sum()),
                unknown_debit=float(debit[u].sum()), gain=float(-loss[k].sum()))
        result[arm] = cells
    return result


def score_block(rec, group, f, a, p, bins, target, valid, cut):
    q = np.repeat(f["frames"], np.diff(f["offsets"]))
    r = prior.base.score_recording(p, PAIRS, target, valid, q, rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
    return dict(r, seed=0, source=group, strata=strata(p, a > 0, bins, target, valid, cut["easy"]))


def fitting(binding, identity, cut, beat, replay):
    train = prior.base.train_source
    dest = PRIVATE/"fitting.json"
    if not dest.exists():
        if replay:
            raise ValueError("Replay may not open labels")
        data = train.fitting_only(identity, beat)
        cfg = run.read(train.source.CONFIG)
        prep = train.prep
        admission = prep.load_admission(ROOT, ROOT/cfg["admission"])
        inventory = run.read(train.bank.old.HOME/"movingdronecrowd_intake_v1/analysis.json")
        meta = {r["clip"]: r for r in inventory["recordings"]}
        lo, rows = 0, []
        for rec in data["recordings"]:
            cell = admission["recordings"][rec]
            clip = meta[cell["clip"]]
            p = ROOT/cell["source_annotation_path"]
            prep.require_fit_record(admission, rec, "fit", p)
            r = prep.mot_record(p.read_bytes(), cell["clip"], range(1, clip["image_frames"]+1))
            times = np.arange(7, r.valid.shape[1]-12)
            counts = np.asarray([r.valid[:, t-7:t+1].all(1).sum() for t in times])
            n = clip["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
            hi = lo+n
            g = cell["exclusion_group"]
            if counts.sum() != n or not np.all(data["sites"][lo:hi] == g):
                raise ValueError("Fit query population mismatch")
            h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
            cv = data["geometry"][lo:hi, 332:356].reshape(-1, 12, 2).astype(float)
            ols = prior.base.motion.slope(h[:, -4:])[:, None]*LEAD
            f = dict(cv=cv, ols=ols, frames=times[counts > 0], offsets=np.r_[0, np.cumsum(counts[counts > 0])])
            a = prior.method.menu(prior.method.backtests(h), np.any(cv != ols, axis=(1, 2)))["three_origins_2"]
            path, bins, _ = paths(f, a)
            rows.append(score_block(rec, g, f, a, path, bins, data["target"][lo:hi], data["valid"][lo:hi], cut))
            lo = hi
        if lo != len(data["ids"]):
            raise ValueError("Fit population mismatch")
        run.once(dest, run.stable(dict(registration=binding, rows=rows)))
    doc = run.read(dest)
    if doc["registration"] != binding:
        raise ValueError("Fit score cache mismatch")
    return doc["rows"]


def aggregate(rows):
    roster = {r["recording"]: r["source"] for r in rows}
    summary = prior.base.grouped_readout(rows, roster, (0,))
    result = {}
    for arm in (RAW, SHARED, MAG):
        out = {}
        for j in range(-1, 3):
            c = [r["strata"][arm][str(j)] for r in rows]
            tot = {k: sum(x[k] for x in c) for k in c[0]}
            tot["known_risk"] = tot["harm"]/tot["reference"] if tot["reference"] else None
            tot["completion_risk"] = (tot["harm"]+tot["unknown_debit"])/tot["reference"] if tot["reference"] else None
            out[str(j)] = tot
        result[arm] = out
    return dict(summary=summary, strata=result)


def score(binding, identity, cut, beat, replay):
    manifest, parent, roster, inventory, meta = prior.target_context()
    frozen = run.read(OUT/"freeze.json")
    if frozen["registration"] != binding or frozen["labels_read"]:
        raise ValueError("Require full frozen path receipt before outcomes")
    cells = run.read(prior.OUT/"freeze.json")["clips"]
    target = []
    for i, (clip, choice, expected) in enumerate(zip(parent["clips"], cells, frozen["clips"], strict=True), 1):
        rec = clip["sequence"]
        dest = PRIVATE/f"score_{rec}.json"
        if not dest.exists():
            if replay:
                raise ValueError("Replay may not open labels")
            receipt, (f, a, p, bins) = forecast_receipt(binding, clip, choice)
            if receipt != expected:
                raise ValueError("New paths differ from their pre-score freeze")
            src = cal.reader(manifest, rec, meta, inventory["provenance"])
            y, valid = cal.label_arrays(src, f)
            row = score_block(rec, roster[rec], f, a, p, bins, y, valid, cut)
            run.once(dest, run.stable(dict(receipt=receipt, row=row)))
        r = run.read(dest)
        if r["receipt"] != expected:
            raise ValueError("Stale target score")
        target.append(r["row"])
        if i % 5 == 0:
            beat(stage="target_score", finished=i)
    s, t = aggregate(fitting(binding, identity, cut, beat, replay)), aggregate(target)
    old = run.read(prior.OUT/"analysis.json")["summary"]
    if t["summary"]["effects"][RAW+"_vs_cv"] != old["effects"][prior.ARM+"_vs_cv"]:
        raise ValueError("Frozen hindcast gain did not reproduce")
    risk = t["summary"]["risk"]
    if risk[RAW] != old["risk"][prior.ARM] or risk[MAG]["switches"] != risk[SHARED]["switches"]:
        raise ValueError("Frozen risk changed or equal-norm control lost support")
    fit = s["summary"]["risk"]
    if fit[RAW]["supported_views"] != 0 or fit[MAG]["switches"] != fit[SHARED]["switches"]:
        raise ValueError("Fitting risk or equal-norm control mismatch")
    result = dict(registration=binding, fitting=s, opened_calibration=t,
        max_matched_velocity_norm_diff=max(x["matched_velocity_norm_max_diff"] for x in frozen["clips"]),
        confirmation_opened=False, risk_rule_changed=False, deployment=False)
    run.once(OUT/"analysis.json", run.stable(result))


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
