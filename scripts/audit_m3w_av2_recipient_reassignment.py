"""Attribute the remaining gap to support, risk and utility on frozen views."""

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_av2_recipient_reassignment as stage


study, base, av = stage.study, stage.base, stage.av


def main():
    key, *_ = stage.register()
    p = stage.OUT/"summary.json"
    summary = study.read(p)
    freeze = study.read(study.checked(summary["freeze"]))
    if freeze["registration"] != key or freeze["held_labels_used"] or summary["reserved_outcomes_opened"]:
        raise ValueError("Need the frozen development readout")
    rows = []
    for name, f in freeze["folds"].items():
        r = study.read(stage.LOCAL/f"{name}.score.json")
        if r["freeze"] != f:
            raise ValueError("Wrong stored score")
        rows.extend(r["rows"])
    opportunities = [r for r in rows if r["capacity"]["reassign_transport"]]
    if len(opportunities) != summary["capacity"]["reassign_transport"]:
        raise ValueError("Changed exact candidate opportunities")
    why = {}
    for k in stage.ARMS:
        c = Counter()
        for r in opportunities:
            a = r["arms"][k]
            tag = "qualified" if a["eligible"] else ("easy_undefined" if not a["easy_defined"] else (
                "defined_risk_failed" if not a["risk"] else "motion_floor_failed"))
            c[tag] += 1
        if c["qualified"] != summary["arms"][k]["eligible"]:
            raise ValueError("Observed choice exceeds transported bank")
        why[k] = {a: c[a] for a in ("qualified", "easy_undefined", "defined_risk_failed", "motion_floor_failed")}
    joint = lambda r, k: r["arms"][k]["eligible"]
    effects = {a+"_minus_"+b: av.interval(rows, lambda r: int(r["capacity"][a])-int(r["capacity"][b]))
               for a, b in (("swap_free", "fixed_free"), ("reassign_free", "swap_free"))}
    out = dict(code=study.artifact(Path(__file__)), source=study.artifact(p), views=len(rows),
        causal_bank_opportunities=len(opportunities), first_failure_in_exact_bank=why,
        union_support_prediction=dict(predicted_but_no_true_easy=sum(r["predicted_easy_available"] and not r["reassign_supported"] for r in rows),
            true_easy_but_not_predicted=sum(r["reassign_supported"] and not r["predicted_easy_available"] for r in rows)),
        joint_changes_vs_additive=dict(rescued=sum(joint(r, "history_joint") and not joint(r, "history_support") for r in rows),
            lost=sum(joint(r, "history_support") and not joint(r, "history_joint") for r in rows)),
        capacity_paired=effects, no_new_label_reads=True, original_2pct=True, reserved_outcomes_opened=False)
    base.immutable(stage.OUT/"attribution.json", out)
    print(study.json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
