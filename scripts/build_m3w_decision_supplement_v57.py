"""Include the held-location learned-map ablation and frozen-weight swap."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v56 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v57"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "opendd_map_path_value_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "head_freeze", "counterfactual_registration", "counterfactual")]
    r, a, h, cr, c = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or a["heads"] != sha(files[2])
            or [v["sha256"] for v in cr["parent"]] != [sha(files[1]), sha(files[2])]
            or c["registration"] != sha(files[3]) or not cr["same_frozen_map_trained_head"]
            or any(v["confirmation_access"] for v in (r, a, cr, c)) or h["held_labels_used_in_fit"]
            or not a["held_location_fully_excluded"] or not a["original_2pct"] or len(h["heads"]) != 63):
        raise ValueError("Require the bound location-held map ablation")
    half = a["summary"]["half"]

    def effect(v):
        lo, hi = v["interval95"]
        return f'{v["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'

    swap = c["summary"]["wrong_map"]
    values = dict(NoMap=effect(half["direct"]["map_vs_no_map"]),
        Wrong=effect(half["direct"]["map_vs_wrong_map"]), Tangent=effect(half["arms"]["map"]["vs_tangent"]),
        Shuffle=effect(half["shuffle"]["map"]), Margin=effect(swap["difference_in_training_assignment_margins"]),
        Swap=effect(swap["paired"]["lower_pp"]), Changed=f'{swap["changed"]:,}', Eligible=f'{swap["eligible"]:,}',
        Pass=half["arms"]["map"]["passes"])
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_learned_road.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\LearnedRoad{k}}}{{{x}}}"
                                              for k, x in values.items())+"\n")
    d["source_bindings"].update({str(p.relative_to(root)): sha(p) for p in (*files, frag)})
    for p in (tex, defs):
        d["files"][p.name] = sha(p)
    (out / "supplement_bindings.json").write_text(json.dumps(d, indent=2, sort_keys=True)+"\n")
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    d = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        package.package(a.output_dir, a.source_zip, d)
