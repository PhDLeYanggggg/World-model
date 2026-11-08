"""Bind the learned two-role stopping experiment and its fixed-deadline control."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v98 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v99"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a, b = timing.legacy.read(timing.CLOCK), timing.read_deadline()
    sha = BASE.cal.digest
    root = PUBLIC.parent/timing.CLOCK
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                       "serialization_recovery_v2", "deadline_registration", "deadline_analysis")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = r"beats the CV-only head but not the frozen neural direction (supplement)."
        new = (r"beats the CV-only head but not the frozen neural direction. Learned timing "
               r"for both roles also fails its same-budget control (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"dual_clock.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    s, t = a["summary"], b["summary"]
    v = dict(Capacity=s["diagnostics"]["nominated_original_budget_capacity"])
    for z, key, tag in ((s, "continuation", "Wait"), (s, "continuation_matched", "WaitMatch"),
                        (t, "deadline", "Deadline"), (t, "deadline_matched", "DeadlineMatch")):
        for f, label in (("edits", "Edits"), ("dual_positive", "Joint"), ("undefined", "Undefined")):
            v[tag+label] = z["metrics"][key][f]
    for tag, e in (("Wait", s["effects"]["continuation_vs_continuation_matched"]),
                   ("Deadline", t["effects"]["deadline_vs_deadline_matched"])):
        lo, hi = e["interval95"]
        v[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\DualClock{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
