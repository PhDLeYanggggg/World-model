"""Bind the double-excluded candidate advantage experiment to the supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v84 as previous
from scripts import build_m3w_decision_manuscript_v79 as packaging

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v85"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    root = PUBLIC.parent / "opendd_candidate_advantage_v1"
    paths = [root / f"{k}.json" for k in ("registration", "analysis", "producer_freeze",
                                          "head_freeze", "capacity_registration", "capacity")]
    r, a, p, h, cr, c = [json.loads(f.read_text()) for f in paths]
    sha = BASE.cal.digest
    key = sha(paths[0])
    if (any(v["registration"] != key for v in (a, p, h)) or cr["parent_registration"] != key
            or a["producers"] != sha(paths[2]) or h["producers"] != sha(paths[2]) or a["heads"] != sha(paths[3])
            or c["registration"] != sha(paths[4]) or not p["outer_sites_excluded"]
            or not c["outcome_aware_only"]
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a, cr, c))):
        raise ValueError("Require double-excluded fitting and original-risk capacity")
    s = a["summary"]
    arms, cap = s["policies"], c["summary"]
    n = arms["hardness"]["edits"]
    if any(v["edits"] != n for k, v in arms.items() if k != "cv"):
        raise ValueError("This table requires matched intervention counts")
    if sum(cap.values()) != arms["hardness"]["views"]:
        raise ValueError("Capacity partition does not match original views")

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    e = s["effects"]
    values = dict(Producers=len(p["heads"]), Heads=len(h["heads"]), Edits=n,
                  Views=arms["hardness"]["views"], GainEffect=effect(e["gain_vs_hardness"]),
                  RiskEffect=effect(e["risk_vs_hardness"]), RiskCV=effect(e["risk_vs_cv"]),
                  GainCV=effect(e["gain_vs_cv"]), RiskGain=effect(e["risk_vs_gain"]),
                  Capacity=cap["qualified"], NoEdit=cap["no_edits"], NoEasy=cap["no_eligible_easy"],
                  Infeasible=cap["risk_infeasible"], NoDual=cap["not_dual_positive"],
                  Unresolved=cap.get("unresolved", 0))
    for k, a in arms.items():
        for label, field in dict(Joint="dual_positive", Risk="risk_pass", Undefined="undefined").items():
            values[k.title()+label] = a[field]
    doc = previous.build(out, style, archive)
    frag = PUBLIC / "advantage.tex"
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Advantage{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(f.relative_to(PUBLIC.parent)): sha(f) for f in (*paths, frag)})
    doc["files"].update({f.name: sha(f) for f in (tex, defs)})
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        packaging.supplement_package.package(a.output_dir, a.source_zip, doc)
