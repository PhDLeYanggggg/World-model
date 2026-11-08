"""Bind the pedestrian candidate-by-budget factorial and its causal comparison."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v107 as previous
from scripts import build_m3w_moving_budget_replay as budget

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v108"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a, b, c = budget.inputs()
    sha = BASE.cal.digest
    paths = [budget.CAPACITY/f"{k}.json" for k in ("registration", "analysis")]
    paths += [budget.POLICY/f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                                 "attribution_registration", "attribution")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = ("On pedestrian development, one-edit support is often feasible against CV, "
               "but rarely beats both motion baselines (supplement).")
        new = ("On pedestrian development, larger budgets expand capacity against OLS4, "
               "but the causal risk screen loses at matched counts (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"moving_budget.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    counts = a["summary"]["counts"]
    vals = dict(Strong=counts["ols_certified_better"])
    for tag, k in (("One", 1), ("Two", 2), ("Four", 4), ("Eight", 8), ("All", "all")):
        for group, label in (("original", "Old"), ("expanded", "Expanded")):
            key = f"{group}_{k}"
            vals[label+tag] = f'{counts.get("strong_"+key, 0)} / {counts[key]}'
    for k, tag in ((1, "One"), (4, "Four"), (8, "Eight")):
        m = b["summary"]["metrics"][f"screen_{k}"]
        vals["Screen"+tag] = f'{m["edits"]:,} / {m["dual_positive"]} / {m["risk_pass"]}'
    e = b["summary"]["effects"]["screen_4_vs_gain"]
    lo, hi = e["interval95"]
    vals["Matched"] = f'{e["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'
    for k, tag in (("observed_pp", "Observed"), ("missing_debit_pp", "Debit"), ("lower_pp", "Lower")):
        vals[tag] = f'{c["summary"]["screen_4_vs_1"][k]["point"]:+.3f}'
    defs.write_text(defs.read_text()+"\n"+"\n".join(
        rf"\newcommand{{\MovingBudget{k}}}{{{x}}}" for k, x in vals.items())+"\n")
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
