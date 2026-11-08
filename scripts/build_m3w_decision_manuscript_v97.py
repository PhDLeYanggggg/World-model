"""Bind the matched support schedule and exact timing/shape bounds."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v96 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v97"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a = timing.legacy.read(timing.SCHEDULE)
    b = timing.read_schedule_audit("capacity_breakdown", "capacity_registration")
    c = timing.read_schedule_audit("shape_capacity", "shape_registration")
    sha = BASE.cal.digest
    paths = [PUBLIC.parent/timing.SCHEDULE/f"{k}.json" for k in (
        "registration", "analysis", "capacity_registration", "capacity_breakdown",
        "shape_registration", "shape_capacity")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = r"risk is not monotone in edit count."
        new = (r"risk is not monotone in edit count. Flexible timing for both roles raises "
               r"same-budget hindsight capacity to \ClockSupportAny{}; an online support-first rule "
               r"does not improve the matched comparison (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"support_schedule.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    s = a["summary"]
    v = dict(Any=s["capacity"]["old_budget_any_order"], Ordered=s["capacity"]["old_budget_support_first"],
             NoReference=s["capacity"]["no_full_easy_reference"], OLSBlocked=b["summary"]["reasons"]["risk_feasible_beats_cv_only"],
             AtMost=s["capacity"]["any_up_to_two"], Shape=c["summary"]["all_menu"])
    for k, tag in (("independent", "Independent"), ("anchor_first", "Anchor"),
                   ("anchor_matched", "Matched"), ("learned_anchor", "Learned")):
        v[tag+"Joint"] = s["metrics"][k]["dual_positive"]
        v[tag+"Edits"] = s["metrics"][k]["edits"]
    e = s["effects"]["anchor_first_vs_anchor_matched"]
    lo, hi = e["interval95"]
    v["Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\ClockSupport{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
