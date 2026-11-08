"""Bind explicit pass-transition supervision and matched intervention controls."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v105 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v106"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    results = timing.legacy.read(timing.PASS_TRANSITION)
    misses = timing.read_support_directions_attribution()
    s = results["summary"]
    sha = BASE.cal.digest
    paths = [PUBLIC.parent/timing.PASS_TRANSITION/f"{f}.json" for f in ("registration", "analysis", "head_freeze")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = ("Broader support directions increase hindsight capacity, but the learned release rule "
               "misses most added opportunities (supplement).")
        new = ("Broader support directions increase hindsight capacity, but direct pass-transition "
               "supervision still loses original passes (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"pass_transition.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    d = s["diagnostics"]
    values = dict(Opportunities=d["opportunities"], Never=d["transition_missed_no_release"],
                  Concentrated=max(sum(r["counts"]["new_opportunities"] for r in rows)
                                   for rows in misses["by_site"].values()))
    for k, tag in (("original", "Original"), ("composite", "Composite"), ("transition", "Transition"),
                   ("rescue_only", "RescueOnly"), ("composite_matched", "Matched"),
                   ("same_row_composite", "SameRow")):
        m = s["metrics"][k]
        values[tag+"Counts"] = f'{m["edits"]} / {m["dual_positive"]} / {m["undefined"]}'
        values[tag+"Transitions"] = f'{d.get(k+"_rescued", 0)} / {d.get(k+"_destroyed", 0)}'
    for k, tag in (("transition_vs_composite_matched", "Matched"), ("transition_vs_original", "Original")):
        e = s["effects"][k]
        lo, hi = e["interval95"]
        values[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(
        rf"\newcommand{{\PassTrans{k}}}{{{x}}}" for k, x in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in paths})
    doc[bindings].update({f.name: sha(f) for f in (tex, defs)})
    (out/receipt).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("main", "supplement"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = a.output_dir or DEFAULT/("build" if a.part == "main" else "supplement")
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == "main":
        packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        packing.supplement_package.package(out, a.source_zip, doc)
