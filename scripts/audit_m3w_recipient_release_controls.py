"""Separate redistribution of an exact total from free-count capacity."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_recipient_release_capacity as base


run, np = base.run, base.np
OUT, PRIVATE = base.OUT, base.PRIVATE
CONTROLS = ("half_free", "curved_total")


def verify_result(val, cv, ols, bank, y, valid, inverse, counts, w, rec, cut):
    if val["status"] == "verified_both":
        r, a = (np.asarray(z, dtype=int) for z in val["witness"])
        actual = base.capacity.evidence(cv, ols, bank, y, valid, inverse, counts, w, rec, cut, r, a)
        if actual != val["evidence"] or not actual["risk"]["supported"]:
            raise ValueError("Control does not pass original risk check")


def effect(rows, arm, reference):
    groups = sorted({r["source"] for r in rows})
    estimate = lambda g: np.mean([
        int(r["states"][arm] == "verified_both")-int(r["states"][reference] == "verified_both")
        for r in rows if r["source"] == g])
    per_group = 100*np.array([estimate(g) for g in groups])
    sample = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    return dict(point=float(per_group.mean()),
                interval95=np.quantile(per_group[sample].mean(1), [.025, .975]).tolist())


def execute(beat, replay):
    binding, identity, cut = base.register()
    analysis = run.read(OUT/"analysis.json")
    if (analysis["registration"] != binding or analysis["calibration_access"]
            or analysis["independent_confirmation"] or len(analysis["recordings"]) != 24):
        raise ValueError("Require the fitting-only capacity analysis")
    meta = run.stable(dict(code=run.artifact(Path(__file__)), capacity=run.artifact(OUT/"analysis.json"),
                           controls=CONTROLS, target_role="original_fitting_only"))
    rows = []
    for i, ((rec, g, cv, ols, bank, y, valid, inv, counts, w), parent) in enumerate(
            zip(base.fitting_blocks(identity, beat), analysis["recordings"], strict=True), 1):
        if parent["recording"] != rec or parent["source"] != g or parent["switches"] != int(counts.sum()):
            raise ValueError("Control recording and original budget changed")
        p = PRIVATE/f"controls_v2_{rec}.json"
        local = run.read(PRIVATE/f"capacity_{rec}.json")
        if local["registration"] != binding:
            raise ValueError("Parent witness binding changed")
        inv_total = np.zeros_like(inv)
        total = np.array([counts.sum()])
        if not p.exists():
            if replay:
                raise ValueError("Replay may not optimize new controls")
            controls = {}
            for key, subset, inverse, count, nested in (
                    ("half_free", bank[:, :4], inv, None, "half_exact"),
                    ("curved_total", bank, inv_total, total, "curved_exact")):
                old = local["menus"][nested]
                current = base.capacity.assess(cv, ols, subset, y, valid, inverse, count, w, rec, cut)
                if old["status"] == "verified_both" and current["status"] != "verified_both":
                    if current["status"] != "unresolved":
                        raise ValueError("Control contradicts smaller verified bank")
                    current = dict(old, inherited=True)
                controls[key] = current
                beat(stage="solve", recording=rec, completed=i-1, arm=key, status=current["status"])
            run.once(p, run.stable(dict(meta=meta, results=controls)))
        row = run.read(p)
        if row["meta"] != meta:
            raise ValueError("Control checkpoint mismatch")
        result = row["results"]
        verify_result(result["half_free"], cv, ols, bank[:, :4], y, valid, inv, None, w, rec, cut)
        verify_result(result["curved_total"], cv, ols, bank, y, valid, inv_total, total, w, rec, cut)
        m = {**parent["menus"], **result}
        if m["curved_free"]["status"] in ("negative_upper", "risk_infeasible") and any(
                m[k]["status"] == "verified_both" for k in CONTROLS):
            raise ValueError("A contained control contradicts the expanded free bound")
        rows.append(dict(recording=rec, source=g, states={k: v["status"] for k, v in m.items()},
                         controls={k: {f: x for f, x in v.items() if f != "witness"}
                                   for k, v in result.items()}))
        beat(stage="checked", recording=rec, completed=i)
    totals = {}
    for k in (*base.ARMS, *CONTROLS):
        c = Counter(r["states"][k] for r in rows)
        totals[k] = {v: c[v] for v in base.capacity.STATES}
    contrasts = {f"{a}_vs_{b}": effect(rows, a, b) for a, b in (
        ("curved_total", "curved_exact"), ("curved_free", "curved_total"),
        ("curved_free", "half_free"), ("half_free", "half_exact"),
    )}
    run.once(OUT/"controls.json", run.stable(dict(meta, summary=totals, rows=rows,
        paired_equal_source_bootstrap_pp=contrasts, resamples=3000,
        outcome_aware_only=True, calibration_access=False, confirmation_access=False, deployment=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    a = cli.parse_args()
    start = time.monotonic()
    with (PRIVATE/"controls.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 1))
            run.kin.prior.source.atomic(PRIVATE/"controls_heartbeat.json", v)
            print(v, flush=True)

        execute(beat, a.replay)


if __name__ == "__main__":
    main()
