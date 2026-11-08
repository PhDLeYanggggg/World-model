"""Bind exact TUMDOT three-reference capacity and teacher ablation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v114 as previous
from scripts import build_m3w_tumdot_joint_replay as replay

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v115"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a, b = replay.inputs()
    doc = previous.build(out, style, archive, part)
    paths = [replay.REPORTS/n/f"{k}.json" for n in replay.NAMES for k in ("analysis", "registration")]
    paths.append(replay.REPORTS/replay.NAMES[1]/"head_freeze.json")
    if part == "main":
        tex = out/"manuscript.tex"
        old = ("On TUMDOT development, paired selection improves local joint passes, "
               "but mean conservative utility still loses to the motion controls (supplement).")
        new = ("On TUMDOT development, a three-reference oracle succeeds at matched radius, "
               "but joint supervision does not improve the causal selector (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"tumdot_teacher.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
        m = a["summary"]["half"]
        macros = dict(Oracle=f'{m["oracle"]["mean_min_pp"]:+.4f}',
                      Pass=str(m["oracle"]["joint"]))
        for key, title in (("complete", "Complete"), ("partial", "Partial"), ("absent", "Absent")):
            d = m["decomposition"]["pair_guard"][key]
            macros[title] = " & ".join(f'{d[k]["point"]:+.4f}' for k in ("observed", "debit", "lower"))
        for key, title in (("joint_vs_local", "Effect"), ("joint_guard_vs_local_guard", "Guard")):
            z = b["summary"]["half"]["effects"][key]["lower"]
            lo, hi = z["interval95"]
            macros[title] = f'{z["point"]:+.4f} pp [{lo:+.4f}, {hi:+.4f}]'
        defs.write_text(defs.read_text()+"\n"+"\n".join(
            rf"\newcommand{{\TumTeacher{k}}}{{{v}}}" for k, v in macros.items())+"\n")
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
