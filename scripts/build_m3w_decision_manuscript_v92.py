"""Add fixed-shape comparisons and a sparse policy that improves on CV."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v91 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v92"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_reserve_shape_v1"
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "direction_registration", "direction")]
    other = PUBLIC.parent/"opendd_reserve_only_v1"
    paths += [other/f"{k}.json" for k in ("registration", "analysis")]
    reg, a, dr, direction, sr, b = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or direction["registration"] != sha(paths[2])
            or b["registration"] != sha(paths[4]) or sr["parent"]["sha256"] != sha(paths[1])
            or [z["sha256"] for z in dr["parents"]] != [sha(paths[1]), sha(paths[5])]
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (reg, a, dr, direction, sr, b))):
        raise ValueError("Require bound development-only shape and sparse policy evidence")

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    e, m = b["summary"]["effects"], b["summary"]["metrics"]
    val = dict(Free=effect(e["reserve_free_vs_cv"]), Single=effect(e["reserve_only_vs_cv"]),
        Combined=effect(e["full_gain_vs_cv"]), Weighted=f'{direction["reserve_free"]["mass_weighted_pp"]:+.3f} pp',
        GainForward=effect(a["summary"]["gain"]["effects"]["ols4_matched_vs_neural_matched"]),
        GainReverse=effect(direction["neural_vs_ols4"]["gain"]),
        IncumbentForward=effect(a["summary"]["incumbent"]["effects"]["ols4_matched_vs_neural_matched"]),
        IncumbentReverse=effect(direction["neural_vs_ols4"]["incumbent"]),
        EasyReverse=effect(direction["neural_vs_ols4"]["easy"]))
    for k, tag in (("gain", "Gain"), ("easy", "Easy"), ("incumbent", "Incumbent")):
        cov = a["summary"][k]["coverage"]
        val[tag+"Matched"], val[tag+"Slots"] = cov["radius_matched"], cov["slots"]
    for k, tag in (("incumbent", "Original"), ("full_gain", "Full"), ("reserve_only", "Only"),
                   ("reserve_free", "Simple"), ("reserve_neural_matched", "Neural"), ("reserve_ols_matched", "OLS")):
        val[tag+"Edits"], val[tag+"Undefined"], val[tag+"Pass"] = (m[k][f] for f in ("edits", "undefined", "dual_positive"))
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"An easy-triggered reserve raises joint passes from \ReserveSlotBasePass{} to "
               r"\ReserveSlotEasyPass{}, but lowers utility over the incumbent; no arm reliably beats CV (supplement).")
        new = (r"Removing incumbent edits gives a one-edit policy with \ShapeSlotFree{} over CV "
               r"but only \ShapeSlotSimplePass{} joint passes (supplement).")
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"shape.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\ShapeSlot{k}}}{{{x}}}" for k, x in val.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in paths})
    doc[bind].update({f.name: sha(f) for f in (tex, defs)})
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
