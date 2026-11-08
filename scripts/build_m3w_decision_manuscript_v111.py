"""Bind low-radius support-exchange witnesses to the manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v110 as previous
from scripts import build_m3w_small_support_exchange_replay as replay

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v111"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
PACK = previous.previous.previous.packing


def build(out, style, archive, part):
    a = replay.inputs()
    doc = previous.build(out, style, archive, part)
    paths = [replay.SOURCE / f"{k}.json" for k in ("registration", "analysis")]
    if part == "main":
        tex = out / "manuscript.tex"
        old = ("On pedestrian development, replacing edited tracks increases missing-label "
               "disagreement, while learning directions at fixed tracks still loses (supplement).")
        new = ("On pedestrian development, low-radius exchanges restore some risk support "
               "but give no joint pass on strong OLS4 views (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        frag = PUBLIC / "support_exchange.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text() + "\n" + old
        receipt, bindings = "supplement_bindings.json", "files"
        counts = a["counts"]
        macros = {}
        for suffix, title in (("one", "One"), ("query", "Query")):
            risk, joint = (counts[f"{key}_{suffix}"] for key in ("risk", "joint"))
            macros[title] = " & ".join(str(v) for v in (risk["witnesses"], risk["undefined_rescues"],
                                                        joint["strictly_joint"], joint["strong_joint"]))
        macros["Strong"] = str(counts["strong_ols"])
        macros["Undefined"] = str(counts["undefined"])
        r = a["intervals"]["joint_query"]
        lo, hi = r["interval95"]
        macros["Effect"] = f'{r["point"]:+.3f} pp [{lo:.3f}, {hi:+.3f}]'
        defs.write_text(defs.read_text() + "\n" + "\n".join(
            rf"\newcommand{{\Exchange{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
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
        PACK.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        PACK.supplement_package.package(out, a.source_zip, doc)
