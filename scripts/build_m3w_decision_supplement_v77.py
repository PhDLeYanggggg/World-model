"""Bind the turn-persistence and exact one-edit capacity evidence."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v76 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v77"
DEFAULT = previous.DEFAULT.parent.parent / PUBLIC.name / "supplement"


def build(out, style, archive):
    src = PUBLIC.parent / "sustained_turn_v2"
    paths = [src / k for k in ("registration.json", "analysis.json",
                              "radius_ceiling_v2/registration.json", "radius_ceiling_v2/analysis.json")]
    r, a, cap_reg, cap = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or cap["registration"] != sha(paths[2])
            or cap_reg["source"]["sha256"] != sha(paths[1]) or cap["independent_confirmation"]
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a, cap_reg))):
        raise ValueError("Require the frozen turn and one-edit evidence")
    s, c = a["summary"], cap["summary"]
    x = s["effects"]["sustained_vs_anchor"]
    lo, hi = x["interval95"]
    expanded = {s["oracle"][k] for k in ("straight", "tapered", "sustained")}
    if len(expanded) != 1 or c["views"] != s["oracle"]["views"]:
        raise ValueError("Unexpected ceiling or comparison roster")
    values = dict(Edits=s["policies"]["anchor"]["edits"], Joint=s["policies"]["sustained"]["dual_positive"],
        Anchor=s["policies"]["anchor"]["dual_positive"], Effect=f'{x["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]',
        Views=c["views"], Legacy=s["oracle"]["anchor"], Expanded=expanded.pop(),
        Bounded=c["bounded"], Unbounded=c["unbounded"])
    doc = previous.build(out, style, archive)
    tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "turn.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Turn{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in (*paths, frag)})
    doc["files"].update({p.name: sha(p) for p in (tex, defs)})
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
        previous.previous.previous.previous.previous.previous.supplement_package.package(a.output_dir, a.source_zip, doc)
