"""Bind radius-matched support directions and missed-opportunity attribution."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v104 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v105"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    stats = timing.legacy.read(timing.SUPPORT_DIRECTIONS)["summary"]
    misses = timing.read_support_directions_attribution()["summary"]
    sha = BASE.cal.digest
    paths = [PUBLIC.parent/timing.SUPPORT_DIRECTIONS/f"{f}.json" for f in
             ("registration", "analysis", "head_freeze", "attribution_registration", "attribution_analysis")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = ("Gain deferral loses to immediate use at matched budgets. A separate reference-loss "
               "guard cannot recover the original joint pass count (supplement).")
        new = ("Broader support directions increase hindsight capacity, but the learned release rule "
               "misses most added opportunities (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"support_directions.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    values = dict(NoRelease=misses["no_release"], WrongQuery=misses["wrong_query"],
                  WrongDirection=misses["wrong_direction"], Realized=misses["realized"],
                  Destroyed=misses["original_joint_destroyed"], Opportunities=misses["new_opportunities"])
    for k, tag in (("original", "Original"), ("neural", "Neural"), ("expanded", "Expanded"),
                   ("same_row_neural", "Matched"), ("trigger_neural", "TriggerNeural"),
                   ("trigger_expanded", "TriggerExpanded"), ("window_neural", "WindowNeural"),
                   ("window_expanded", "WindowExpanded")):
        m = stats["metrics"][k]
        values[tag+"Counts"] = f'{m["edits"]} / {m["dual_positive"]} / {m["undefined"]}'
    for k, tag in (("expanded_vs_same_row_neural", "Matched"), ("expanded_vs_original", "Original")):
        e = stats["effects"][k]
        lo, hi = e["interval95"]
        values[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(
        rf"\newcommand{{\SupportDir{k}}}{{{x}}}" for k, x in values.items())+"\n")
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
