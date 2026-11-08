"""Determine whether changing recipients or path geometry can fix fitting risk."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_past_sampling_audit as prior
from src.evaluation import m3w_recipient_release_capacity as capacity
from src.world_model import m3w_released_recipient_paths as path_model


run, np = prior.run, prior.np
OUT = prior.OUT.parent/"recipient_release_capacity_v1"
PRIVATE = prior.PRIVATE.parent/OUT.name
ARMS = path_model.MENUS


def register():
    parent, identity, cut = prior.register()
    p = OUT/"registration.json"
    cap = prior.OUT/"convex_capacity.json"
    if run.read(cap)["excluded_by_optimistic_bound"] != 24:
        raise ValueError("The prior fixed-recipient certificate is required")
    run.once(p, run.stable(dict(
        code=run.artifact(Path(__file__)), optimizer=run.artifact(Path(capacity.__file__)),
        shared_solver=run.artifact(Path(capacity.solve.__code__.co_filename)),
        paths=run.artifact(Path(path_model.__file__)), parent=run.artifact(cap),
        previous_registration=parent, menus=dict(zip(ARMS, path_model.WIDTHS, strict=True)),
        original_query_counts_in_exact_arms=True, final_arm_unrestricted_counts=True,
        conservative_references=["CV", "OLS4"], risk_budget=.02,
        solver_time_limits_seconds=[2., 8.], scope="original_fitting_labels_hindsight_only",
        calibration_access=False, confirmation_access=False, deployment=False)))
    return run.sha(p), identity, cut


def fitting_blocks(identity, beat):
    train = prior.prior.base.train_source
    data = train.fitting_only(identity, beat)
    prep = train.prep
    admission = prep.load_admission(ROOT, ROOT/run.read(train.source.CONFIG)["admission"])
    support = run.read(train.bank.old.HOME/"movingdronecrowd_intake_v1/analysis.json")
    meta = {r["clip"]: r for r in support["recordings"]}
    lo = 0
    for rec in data["recordings"]:
        cell = admission["recordings"][rec]
        clip = meta[cell["clip"]]
        p = ROOT/cell["source_annotation_path"]
        prep.require_fit_record(admission, rec, "fit", p)
        raw = prep.mot_record(p.read_bytes(), cell["clip"], range(1, clip["image_frames"]+1))
        sizes = np.array([raw.valid[:, t-7:t+1].all(1).sum() for t in range(7, raw.valid.shape[1]-12)])
        sizes = sizes[sizes > 0]
        n = clip["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
        hi = lo+n
        group = cell["exclusion_group"]
        if sizes.sum() != n or not np.all(data["sites"][lo:hi] == group):
            raise ValueError("Fitting query population changed")
        h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
        cv = data["geometry"][lo:hi, 332:356].reshape(-1, 12, 2).astype(float)
        ols = prior.prior.base.motion.slope(h[:, -4:])[:, None]*prior.previous.LEAD
        legal = np.any(ols != cv, axis=(1, 2))
        on = prior.prior.method.menu(prior.prior.method.backtests(h), legal)["three_origins_2"] > 0
        inv = np.repeat(np.arange(len(sizes)), sizes)
        counts = np.bincount(inv[on], minlength=len(sizes))
        bank = path_model.paths(h, cv, ols)
        yield rec, group, cv, ols, bank, data["target"][lo:hi], data["valid"][lo:hi], inv, counts, 1./sizes[inv]
        lo = hi
    if lo != len(data["ids"]) or lo != 69746:
        raise ValueError("Original fitting roster required")


def summarize(blocks):
    stats = {}
    for arm in ARMS:
        counts = Counter(row["menus"][arm]["status"] for row in blocks)
        stats[arm] = {k: counts[k] for k in capacity.STATES}
        wins = [row for row in blocks if row["menus"][arm]["status"] == "verified_both"]
        stats[arm]["positive_groups"] = len({r["source"] for r in wins})
        stats[arm]["positive_switches"] = sum(row["menus"][arm]["evidence"]["switched"] for row in wins)
    transitions = {}
    for a, b in zip(ARMS[:-1], ARMS[1:], strict=True):
        transitions[b] = sum(r["menus"][a]["status"] != "verified_both"
                             and r["menus"][b]["status"] == "verified_both" for r in blocks)
    return dict(capacity=stats, newly_verified=transitions, recordings=len(blocks),
                source_groups=len({r["source"] for r in blocks}),
                exact_switches=sum(r["switches"] for r in blocks))


def execute(binding, identity, cut, beat, replay):
    rows = []
    for i, (rec, g, cv, ols, bank, y, valid, inv, counts, w) in enumerate(fitting_blocks(identity, beat), 1):
        p = PRIVATE/f"capacity_{rec}.json"
        if not p.exists():
            if replay:
                raise ValueError("Replay may not rerun optimization")
            values = {}
            for arm, width in zip(ARMS, path_model.WIDTHS, strict=True):
                budget = None if arm == ARMS[-1] else counts
                val = capacity.assess(cv, ols, bank[:, :width], y, valid, inv, budget, w, rec, cut)
                prev = next(reversed(values.values())) if values else None
                if prev and prev["status"] == "verified_both" and val["status"] != "verified_both":
                    if val["status"] != "unresolved":
                        raise ValueError("Nested bank excludes a verified smaller witness")
                    val = dict(prev, inherited=True)
                values[arm] = val
                beat(stage="solve", recording=rec, finished=i-1, arm=arm, status=val["status"])
            run.once(p, run.stable(dict(registration=binding, recording=rec, source=g,
                switches=int(counts.sum()), menus=values)))
        row = run.read(p)
        if (row["registration"] != binding or row["recording"] != rec or row["source"] != g
                or row["switches"] != int(counts.sum())):
            raise ValueError("Recording capacity checkpoint changed")
        public = {}
        for arm, width in zip(ARMS, path_model.WIDTHS, strict=True):
            r = row["menus"][arm]
            if r["status"] == "verified_both":
                a, b = (np.asarray(z, dtype=int) for z in r["witness"])
                budget = None if arm == ARMS[-1] else counts
                check = capacity.evidence(cv, ols, bank[:, :width], y, valid, inv, budget, w, rec, cut, a, b)
                if check != r["evidence"] or not check["risk"]["supported"]:
                    raise ValueError("Original risk check or witness replay failed")
            public[arm] = {k: v for k, v in r.items() if k != "witness"}
        rows.append(dict(recording=rec, source=g, switches=row["switches"], menus=public))
        beat(stage="witness_check", recording=rec, finished=i)
    summary = summarize(rows)
    if (summary["recordings"], summary["source_groups"], summary["exact_switches"]) != (24, 9, 31923):
        raise ValueError("Original intervention population changed")
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, summary=summary, recordings=rows,
        outcome_aware_only=True, original_risk_rule=True, calibration_access=False,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "score", "replay"))
    a = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut = register()
        start = time.monotonic()

        def beat(**v):
            v.update(pid=os.getpid(), phase=a.phase, seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            print(v, flush=True)

        if a.phase != "register":
            execute(binding, identity, cut, beat, a.phase == "replay")


if __name__ == "__main__":
    main()
