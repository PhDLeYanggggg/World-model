"""Bind the OpenDD component and difficulty readouts to the supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v83 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v84"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    aroot, broot = [PUBLIC.parent / f"opendd_{k}_v1" for k in ("component_guard", "hardness_gate")]
    ap, ar, bp, br, bh = [root / filename for root, filename in (
        (aroot, "analysis.json"), (aroot, "registration.json"), (broot, "analysis.json"),
        (broot, "registration.json"), (broot, "head_freeze.json"))]
    sha = BASE.cal.digest
    a, r0, b, r1, h = [json.loads(p.read_text()) for p in (ap, ar, bp, br, bh)]
    if (a["registration"] != sha(ar) or any(v["registration"] != sha(br) for v in (b, h))
            or b["heads"] != sha(bh) or r1["parent"]["sha256"] != sha(ap)
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (a, b, r0, r1))):
        raise ValueError("Require the frozen development diagnostics")
    a, b = a["summary"], b["summary"]
    old, guard = [a["matched"][k] for k in ("prefix_match", "guard")]
    if any(guard[f] != old[f] for f in ("edits", "views", "undefined")):
        raise ValueError("Component guard must preserve edit budget and support")
    if any(b["raw"]["gated"][f] != b["raw"]["matched_motion"][f] for f in ("edits", "views")):
        raise ValueError("Require equal counts in the decision contrast")

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    v = dict(Fits=len(h["heads"]), Views=old["views"], Edits=old["edits"],
        Guard=guard["dual_positive"], Prefix=old["dual_positive"], Gate=b["slot"]["gated"]["dual_positive"],
        RawEdits=b["raw"]["gated"]["edits"], Direction=effect(a["raw_effects"]["guard_vs_equal_peak"]),
        GuardMatched=effect(a["matched_effects"]["guard_vs_prefix_match"]),
        GatedMotion=effect(b["effects"]["raw_gated_vs_motion"]), GatedCV=effect(b["effects"]["raw_gated_vs_cv"]),
        GatedFixed=effect(b["effects"]["fixed_slot"]), Swaps=b["shape_switches"],
        AUROC=f'{b["diagnostics"]["hard"]["auroc"]:.3f}', MotionAUC=f'{b["diagnostics"]["motion"]["auroc"]:.3f}')
    for name, k in dict(Easy="easy", Hard="hard", Other="other_observed", Unobserved="no_label").items():
        v[name+"Effect"] = effect(a["raw"]["full"]["strata"][k])
        v[name+"Rows"] = a["unique_slice_rows"][k]
    doc = previous.build(out, style, archive)
    frag = PUBLIC / "difficulty.tex"
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Difficulty{k}}}{{{value}}}" for k, value in v.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in (ap, ar, bp, br, bh, frag)})
    doc["files"].update({p.name: sha(p) for p in (tex, defs)})
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
        previous.packaging.supplement_package.package(a.output_dir, a.source_zip, doc)
