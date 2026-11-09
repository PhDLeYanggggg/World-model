"""Score or replay frozen relative projections using the corrected summary key."""

import argparse
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_relative_projection as run


def collect(open_outcomes):
    key, data, src, _, _, outer, control, _, _, parent = run.register()
    study, arms = run.study, run.ARMS
    p = run.OUT/"freeze.json"
    seal = study.read(p)
    cp = run.OUT/"calibration.json"
    if (seal["registration"] != key
            or any(seal[k] for k in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened"))):
        raise ValueError("Require forecasts frozen before calibration and held scoring")
    if not cp.exists() and not open_outcomes:
        raise ValueError("Replay requires a calibration receipt")
    results = {}
    rules_by_fold = {}
    for role in ("calibration", "held"):
        rows = []
        for name, ref in seal["folds"].items():
            f = study.read(study.checked(ref))
            mark = run.LOCAL/f"{name}.{role}.score.json"
            if not mark.exists():
                if not open_outcomes:
                    raise ValueError("Replay may not read outcomes")
                run.base.immutable(mark, dict(freeze=ref, role=role,
                    rows=run.fold_score(data, src, f, control["folds"][name], outer["folds"][name], role)))
            receipt = study.read(mark)
            if (receipt["freeze"] != ref or receipt["role"] != role
                    or [r["clip"] for r in receipt["rows"]] != [r["clip"] for r in f["rows"][role]]
                    or any(r["city"] != f["identity"][role] for r in receipt["rows"])):
                raise ValueError("Wrong scored receipt")
            if role == "calibration":
                rules_by_fold[name] = {k: run.timing.old.action.prior.calibration([dict(score=r["arms"][k]["confidence"],
                    strict=r["arms"][k]["strict"], add_ok=True) for r in receipt["rows"]]) for k in arms[2:]}
            else:
                for k in arms[2:]:
                    for r in receipt["rows"]:
                        t = rules_by_fold[name][k]["threshold"]
                        r["arms"][k]["release"] = t is not None and r["arms"][k]["confidence"] >= t
            rows.extend(receipt["rows"])
        if role == "calibration":
            run.base.immutable(cp, dict(freeze=study.artifact(p), rules=rules_by_fold, held_labels_used=False))
        results[role] = dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows),
            arms=run.summarize(rows), by_city={c: {k: sum(r["arms"][k]["eligible"] for r in rows if r["city"] == c) for k in arms}
                for c in sorted({r["city"] for r in rows})})
        if role == "held":
            results[role]["releases"] = {k: dict(calibration_pass=sum(f[k]["accepted"] for f in rules_by_fold.values()),
                selected=sum(r["arms"][k]["release"] for r in rows),
                strict=sum(r["arms"][k]["release"] and r["arms"][k]["strict"] for r in rows)) for k in arms[2:]}
    cal_score = results["calibration"]
    if cal_score["views"] != parent["views"] or cal_score["arms"]["frozen_position"]["eligible"] != parent["arms"]["raw_projection"]["eligible"]:
        raise ValueError("Frozen projected reference did not replay")
    run.base.immutable(run.OUT/"summary.json", dict(**results, freeze=study.artifact(p), calibration_receipt=study.artifact(cp),
        result_source="fresh_run", packaging="scored receipt replay; corrected calibration key only", independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False))
    print(study.json.dumps({k: dict(views=v["views"], eligible={a: z["eligible"] for a, z in v["arms"].items()})
                           for k, v in results.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--score", action="store_true", help="Read outcomes only to fill missing role-specific scoring receipts")
    args = p.parse_args()
    run.torch.set_num_threads(4)
    run.torch.set_num_interop_threads(1)
    with (run.LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(args.score)


if __name__ == "__main__":
    main()
