"""Update the main paper with controlled trajectory-shape evidence."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v13 as main_package
from scripts import build_m3w_decision_manuscript_v66 as main
from scripts import build_m3w_decision_supplement_v20 as supplement_package
from scripts import build_m3w_decision_supplement_v69 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v70"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive, part):
    home = PUBLIC.parent
    cap = home / "radius_matched_turn_menu_v1"
    value = home / "shape_menu_value_v1"
    paths = [cap / f"{k}.json" for k in ("registration", "analysis")]+[value / f"{k}.json" for k in (
        "registration", "analysis", "head_freeze", "tangent_control")]
    cr, c, vr, a, h, t = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (c["registration"] != sha(paths[0]) or vr["prior"]["sha256"] != sha(paths[1])
            or any(x["registration"] != sha(paths[2]) for x in (a, h, t)) or a["heads"] != sha(paths[4])
            or t["analysis"]["sha256"] != sha(paths[3]) or len(h["heads"]) != 42
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (cr, c, vr, a, t))
            or not a["exact_rows_and_radii"] or not t["exact_rows_and_radii"] or not c["oracle_label_aware_only"]):
        raise ValueError("Require bound same-radius causal and hindsight evidence")
    s, z, u = c["summary"], a["summary"], t["summary"]
    if len({v["edits"] for v in (*s["arms"].values(), *z["arms"].values(), u["tangent"])}) != 1:
        raise ValueError("Recorded controls do not share one edit roster")

    def effect(q):
        low, high = q["interval95"]
        return f'{q["point"]:+.3f} pp [{low:+.3f}, {high:+.3f}]'

    v = dict(Edits=f'{z["arms"]["anchor"]["edits"]:,}', Views=z["arms"]["anchor"]["views"],
        Undefined=z["arms"]["anchor"]["undefined"], Timing=s["capacity"]["timing"]["verified_positive_both"],
        Full=s["capacity"]["full"]["verified_positive_both"], TimingEffect=effect(s["direct"]["full_prefix_vs_timing_prefix"]),
        Linear=effect(z["direct"]["learned_dynamics_vs_anchor"]), Prefix=effect(z["direct"]["learned_dynamics_vs_full_prefix"]),
        Tangent=effect(u["direct"]["dynamics_vs_tangent"]),
        DynPass=z["arms"]["learned_dynamics"]["dual_positive"], LinearPass=z["arms"]["anchor"]["dual_positive"],
        PrefixPass=z["arms"]["full_prefix"]["dual_positive"], TangentPass=u["tangent"]["dual_positive"],
        ValuePass=z["arms"]["learned_gain"]["dual_positive"], ValueRiskPass=z["arms"]["learned_risk"]["dual_positive"])
    values = "\n".join(rf"\newcommand{{\RadiusShape{k}}}{{{x}}}" for k, x in v.items())+"\n"
    if part == "main":
        doc = main.build(out, style, archive, "main")
        tex, defs = out / "manuscript.tex", out / "direct_count_values.tex"
        frag = PUBLIC / "main_shape.tex"
        old = "Learning direct\ngain and harm shows some discrimination, but adding peer histories\ndoes not improve matched utility."
        new = ("At fixed recipients and displacement radii, learned shape choice\n"
               "improves over a linear edit but does not establish a gain over\n"
               "a simple observed-prefix rule.")
        content = BASE.replace_once(tex.read_text(), old, new)
        marker = r"\section{Limitations and Independent Evidence}"
        bindings, manifest = "build_bindings", "source_bindings.json"
    else:
        doc = previous.build(out, style, archive)
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        frag = PUBLIC / "supplement_shape.tex"
        content = tex.read_text()
        marker = r"\section*{LLM Usage Statement}"
        bindings, manifest = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(content, marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+values)
    doc["source_bindings"].update({str(p.relative_to(home)): sha(p) for p in (*paths, frag)})
    for p in (tex, defs):
        doc[bindings][p.name] = sha(p)
    (out / manifest).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
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
        main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        supplement_package.package(out, a.source_zip, doc)
