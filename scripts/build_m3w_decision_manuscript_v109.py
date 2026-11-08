"""Bind the matched pedestrian observability comparison to the draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v108 as previous
from scripts import build_m3w_moving_observability_replay as replay

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v109"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    summary = replay.inputs()["summary"]
    doc = previous.build(out, style, archive, part)
    paths = [replay.SOURCE / f"{key}.json" for key in ("registration", "analysis", "head_freeze")]
    if part == "main":
        tex = out / "manuscript.tex"
        old = ("On pedestrian development, larger budgets expand capacity against OLS4, "
               "but the causal risk screen loses at matched counts (supplement).")
        new = ("On pedestrian development, wider budgets expand hindsight capacity, "
               "but better missingness prediction does not improve matched policy utility (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex = out / "supplement.tex"
        frag = PUBLIC / "observability.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        defs = out / "supplement_values.tex"
        macros = {}
        for key, name in (("entangled", "Entangled"), ("gain_rank", "Gain"),
                          ("no_missing", "NoMissing"), ("motion", "Motion"),
                          ("geometry", "Geometry"), ("geometry_free", "Free")):
            m = summary["metrics"][key]
            macros[name] = " & ".join(f"{m[k]:,}" for k in
                                      ("edits", "unknown_edits", "dual_positive", "risk_pass"))
        for key, name in (("geometry_vs_entangled", "Effect"), ("geometry_free_vs_entangled", "FreeEffect")):
            m = summary["effects"][key]
            lo, hi = m["interval95"]
            macros[name] = f'{m["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'
        for key, name in (("motion", "MotionBrier"), ("geometry", "GeometryBrier"),
                          ("geometry_minus_motion", "BrierDifference")):
            m = summary["brier"][key]
            lo, hi = m["interval95"]
            macros[name] = (f'{m["point"]:+.5f} [{lo:+.5f}, {hi:+.5f}]' if name == "BrierDifference"
                            else f'{m["point"]:.5f}')
        defs.write_text(defs.read_text() + "\n" + "\n".join(
            rf"\newcommand{{\Obs{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
        new = frag.read_text() + "\n\n" + old
        receipt, bindings = "supplement_bindings.json", "files"
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
        previous.packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        previous.packing.supplement_package.package(out, a.source_zip, doc)
