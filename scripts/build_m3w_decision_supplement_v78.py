"""Bind the matched two-edit control and exact finite-bank ceiling."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v77 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v78"
DEFAULT = previous.DEFAULT.parent.parent / PUBLIC.name / "supplement"


def build(out, style, archive):
    src = PUBLIC.parent / "two_edit_v1"
    paths = [src / f"{k}.json" for k in ("registration", "analysis", "head_freeze", "readout_repair_v2")]
    r, a, heads, fix = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (any(v["registration"] != sha(paths[0]) for v in (a, heads, fix)) or a["heads"] != sha(paths[2])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a))):
        raise ValueError("Require the frozen matched study")
    s = a["summary"]
    full, cap = s["full"], s["oracle"]
    edits = {v["edits"] for v in full.values()}
    if len(edits) != 1:
        raise ValueError("Expected identical realized edit counts")
    x = s["paired_gain_vs_independent"]["joint"]
    lo, hi = x["interval95"]
    values = dict(One=cap["one"], Any=cap["one_or_pair"], Views=cap["views"], Same=cap["same_query"],
        Cross=cap["separate_queries"], Edits=edits.pop(), Joint=full["joint"]["dual_positive"],
        Independent=full["independent"]["dual_positive"], Gain=f'{x["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    doc = previous.build(out, style, archive)
    tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "two_edit.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\TwoEdit{k}}}{{{v}}}" for k, v in values.items())+"\n")
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
        previous.previous.previous.previous.previous.previous.previous.supplement_package.package(a.output_dir, a.source_zip, doc)
