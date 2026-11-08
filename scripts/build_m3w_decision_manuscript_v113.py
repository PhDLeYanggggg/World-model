"""Bind fixed-radius first-offer ranking and shortlist recall."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v112 as previous
from scripts import build_m3w_opening_slate_replay as replay

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v113"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a = replay.inputs()
    doc = previous.build(out, style, archive, part)
    paths = [replay.SOURCE/f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    if part == "main":
        tex = out/"manuscript.tex"
        old = ("On pedestrian development, same-budget first-edit timing adds no reliable utility "
               "and a second offer adds no hindsight coverage (supplement).")
        new = ("On pedestrian development, diverse shortlists improve some recall, "
               "but listwise training does not improve the first edit (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"opening_slate.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
        recall = a["recall"]
        macros = {name: " & ".join(str(recall[f"{key}_{c}"]) for c in
                                  ("flat_1", "flat_8", "row_8", "direction_8", "row_16"))
                  for key, name in (("frozen", "Frozen"), ("ridge", "Ridge"),
                                    ("pointwise", "Point"), ("listwise", "List"))}
        r = a["effects"]["listwise_vs_pointwise"]
        lo, hi = r["interval95"]
        macros["Effect"] = f'{r["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
        macros["All"] = str(recall["all"])
        macros["Empty"] = str(sum(not r["fit_sources_with_offers"] for rs in a["by_site"].values() for r in rs))
        defs.write_text(defs.read_text()+"\n"+"\n".join(
            rf"\newcommand{{\Slate{name}}}{{{val}}}" for name, val in macros.items())+"\n")
        doc[bindings][defs.name] = BASE.cal.digest(defs)
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): BASE.cal.digest(p) for p in paths})
    doc[bindings][tex.name] = BASE.cal.digest(tex)
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
        PACK.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        PACK.supplement_package.package(out, a.source_zip, doc)
