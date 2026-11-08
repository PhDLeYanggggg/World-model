"""Bind the protected-gain experiment and the fixed-support timing ceiling."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v100 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v101"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a, b = timing.legacy.read(timing.PROTECTED_GAIN), timing.read_protected_clock()
    sha = BASE.cal.digest
    root = PUBLIC.parent/timing.PROTECTED_GAIN
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                       "clock_registration", "clock_capacity")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = ("Neither learned timing nor matched-budget slate ranking beats its reference policy "
               "on both utility and risk (supplement).")
        new = ("Protecting the support edit reduces reference loss but does not improve utility "
               "and risk over the original policy (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"protected_gain.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    s, c = a["summary"], b["summary"]
    d = s["diagnostics"]
    v = dict(GainLost=d["gain_only_lost_reference"], ProtectedLost=d["protected_lost_reference"],
             ProtectedNew=d["protected_new_reference"], GuardChanges=d["guard_changes"],
             Seen=d["support_seen"], Queries=d["gain_queries"], Current=c["current_protected"],
             Delayed=c["delay_reference_preserved"], Unrestricted=c["delay_any"])
    for k, tag in (("original", "Original"), ("gain_only", "Gain"),
                   ("protected", "Protected"), ("protected_masked", "Masked"), ("guard", "Guard")):
        m = s["metrics"][k]
        v[tag+"Counts"] = f'{m["dual_positive"]} / {m["risk_pass"]} / {m["undefined"]}'
    for k, tag in (("protected_vs_original", "Primary"), ("guard_vs_original", "Guard"),
                   ("protected_vs_protected_masked", "Context")):
        e = s["effects"][k]
        lo, hi = e["interval95"]
        v[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\ProtectedGain{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
