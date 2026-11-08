"""Bind the bounded support stopping and recipient-menu experiment."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v93 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v94"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a, menu = timing.legacy.read(timing.STOP), timing.read_menu()
    sha = BASE.cal.digest
    folder = PUBLIC.parent/timing.STOP
    paths = [folder/f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                         "menu_registration", "menu_capacity")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = r"at lower utility (supplement)."
        new = r"at lower utility. Learning to delay the support slot does not reliably help (supplement)."
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"support_stop.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    v = {}
    for key, tag in (("continue_vs_continue_matched", "Matched"), ("continue_vs_cv", "CV")):
        e = a["summary"]["effects"][key]
        lo, hi = e["interval95"]
        v[tag] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    metrics = a["summary"]["metrics"]
    for key, tag in (("immediate", "Immediate"), ("continue", "Continue"),
                     ("continue_matched", "Replay")):
        for f, label in (("edits", "Edits"), ("undefined", "Undefined"), ("dual_positive", "Joint")):
            v[tag+label] = metrics[key][f]
    for clock, tag in (("fixed", "Fixed"), ("plus_two", "Two"),
                       ("after_release", "Later"), ("any_query", "Any")):
        for key, label in (("top_easy", "Top"), ("all_predicted_easy", "Easy"), ("all_legal", "Legal")):
            v[tag+label] = menu["summary"][clock][key]
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\StopSupport{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
