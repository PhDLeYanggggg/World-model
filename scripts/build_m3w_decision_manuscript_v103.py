"""Bind two matched early-support studies to the working manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v102 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v103"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    local, record = (timing.legacy.read(k) for k in (timing.EARLY_REFERENCE, timing.EARLY_ADVANTAGE))
    sha = BASE.cal.digest
    paths = [PUBLIC.parent/k/f"{f}.json" for k in (timing.EARLY_REFERENCE, timing.EARLY_ADVANTAGE)
             for f in ("registration", "analysis", "head_freeze")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = ("Protecting support reduces reference loss, but waiting for a replacement gain "
               "edit loses to immediate use at matched budgets (supplement).")
        new = ("Gain deferral loses to immediate use at matched budgets. Earlier support "
               "increases interventions but can lose easy references (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"early_reference.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    a, b = local["summary"], record["summary"]
    d = b["diagnostics"]
    vals = dict(Added=d["mass_new"], Advanced=d["mass_advance"], Lost=d["mass_lost_reference"],
                MatchedChanged=d["easy_matched_recipient_changes"], Oracle=d["oracle_advanced"]+d["oracle_recovered_missing"])
    for s, k, tag in ((a, "original", "Original"), (a, "mass", "Local"),
                      (b, "mass", "Record"), (b, "incidence", "Utility"),
                      (b, "easy_matched", "Matched"), (b, "early_oracle", "Ceiling")):
        m = s["metrics"][k]
        vals[tag+"Counts"] = f'{m["edits"]} / {m["dual_positive"]} / {m["undefined"]}'
    for k, tag in (("mass_vs_original", "Primary"), ("mass_vs_easy_matched", "Matched")):
        e = b["effects"][k]
        lo, hi = e["interval95"]
        vals[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(
        rf"\newcommand{{\EarlyRef{k}}}{{{x}}}" for k, x in vals.items())+"\n")
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
