"""Bind the matched-recipient ablation to the evidence supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v70 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v71"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name / "supplement"
BASE = previous.BASE


def build(out, style, archive):
    study = PUBLIC.parent / "recipient_slot_v1"
    paths = [study / f"{name}.json" for name in ("registration", "head_freeze", "analysis")]
    r, h, a = [json.loads(p.read_text()) for p in paths]
    digest = BASE.cal.digest
    if (h["registration"] != digest(paths[0]) or a["registration"] != digest(paths[0])
            or a["heads"] != digest(paths[1]) or r["confirmation_access"] or a["confirmation_access"]
            or not r["original_2pct"] or not a["original_2pct"] or not a["query_radii_exact"]):
        raise ValueError("Require bound fixed-radius recipient evidence")
    s = a["summary"]
    old, new = (s["arms"][name] for name in ("frozen_prefix", "joint_prefix"))
    cap = s["oracle"]
    if sum(cap.values()) != old["views"] or new["edits"] != old["edits"]:
        raise ValueError("Recipient controls do not share a single roster of edit slots")
    g = s["direct"]["joint_prefix_vs_frozen_prefix"]
    lo, hi = g["interval95"]
    values = dict(Edits=f'{old["edits"]:,}', Oracle=cap["verified_positive_both"], Views=old["views"],
        Frozen=old["dual_positive"], Blocked=sum(cap[k] for k in ("zero_edits", "no_eligible_easy", "absent_easy")),
        Empty=cap["zero_edits"], Ineligible=cap["no_eligible_easy"], Fits=len(h["heads"]),
        Learned=new["dual_positive"], Undefined=new["undefined"], Gain=f'{g["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]')
    doc = previous.build(out, style, archive, "supplement")
    tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "recipient_slots.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Slot{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): digest(p) for p in (*paths, frag)})
    doc["files"].update({p.name: digest(p) for p in (tex, defs)})
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        previous.supplement_package.package(a.output_dir, a.source_zip, doc)
