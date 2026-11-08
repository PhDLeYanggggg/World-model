"""Bind exact missingness attribution and fixed-recipient direction results."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v109 as previous
from scripts import build_m3w_incumbent_direction_replay as replay

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v110"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    result, attr = replay.inputs()
    doc = previous.build(out, style, archive, part)
    paths = [root / f"{k}.json" for root, names in
             ((replay.SOURCE, ("registration", "analysis", "head_freeze")),
              (replay.ATTR, ("registration", "analysis"))) for k in names]
    if part == "main":
        tex = out / "manuscript.tex"
        old = ("On pedestrian development, wider budgets expand hindsight capacity, "
               "but better missingness prediction does not improve matched policy utility (supplement).")
        new = ("On pedestrian development, replacing edited tracks increases missing-label "
               "disagreement, while learning directions at fixed tracks still loses (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        frag = PUBLIC / "incumbent_direction.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text() + "\n" + old
        receipt, bindings = "supplement_bindings.json", "files"
        macros = {}
        for key, name in (("incumbent", "Original"), ("direct", "Direct"),
                          ("absolute_matched", "Absolute"), ("no_debit_matched", "NoDebit"),
                          ("oracle", "Oracle")):
            m = result["summary"]["metrics"][key]
            val = result["summary"]["effects"].get(key, {}).get("lower")
            text = f'{val["point"]:+.3f}' if val else "0"
            macros[name] = f'{m["swaps"]:,} & {m["dual_positive"]} & {text}'
        for k, name in (("observed", "Observed"), ("debit", "Debit")):
            macros[name] = f'{attr["summary"]["geometry"][k]["point"]:+.3f}'
        m = result["summary"]["effects"]["direct"]["lower"]
        lo, hi = m["interval95"]
        macros["DirectCI"] = f'[{lo:+.3f}, {hi:+.3f}]'
        defs.write_text(defs.read_text() + "\n" + "\n".join(
            rf"\newcommand{{\Dir{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
        doc[bindings][defs.name] = BASE.cal.digest(defs)
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): BASE.cal.digest(p) for p in paths})
    doc[bindings][tex.name] = BASE.cal.digest(tex)
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("main", "supplement"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = a.output_dir or DEFAULT / ("build" if a.part == "main" else "supplement")
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == "main":
        previous.previous.packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        previous.previous.packing.supplement_package.package(out, a.source_zip, doc)
