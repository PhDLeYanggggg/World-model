"""Add the broad-motion experiment and predicted-feasibility repair."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v37 as previous

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v38"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    dense = home/"dense_motion_calibration_v1"
    repaired = home/"dense_safe_union_v1"
    sources = [dense/"analysis.json", repaired/"analysis.json", repaired/"verification.json",
               dense/"predicted_feasibility.json", dense/"registration.json", repaired/"registration.json"]
    d, u, v, a = [json.loads(f.read_text()) for f in sources[:4]]
    if (v["dense_analysis_sha256"] != BASE.cal.digest(sources[0])
            or v["union_analysis_sha256"] != BASE.cal.digest(sources[1])
            or v["same_query_amplitude_verified"] != 33720 or v["confirmation_access"]
            or a["positive_gain_singleton_views"] != 120 or a["joint_empty_views"] != 120
            or not d["original_risk_unchanged"] or u["risk_rule_changed"] or not u["no_new_fits"]):
        raise ValueError("Require exact matched readout and unchanged risk")
    x, y = d["summary"], u["summary"]
    rows = []
    for s, key, title in (
        (x, "history_ols4_velocity", "OLS4"),
        (x, "fixed_half", "Fixed half"),
        (x, "independent", "Learned gain"),
        (x, "joint", "Price ladder"),
        (y, "union", "Feasible union"),
        (y, "matched", "Matched hindcast"),
    ):
        effect = s["effects"][key+"_vs_cv"]["completion_lower_pp"]["point"]
        rows.append(f'{title} & {effect:+.3f} & {s["risk"][key]["supported_views"]} '+r'\\')
    contrast = y["effects"]["union_vs_matched"]["completion_lower_pp"]
    values = {
        "Rows": "\n".join(rows), "UnionPass": y["risk"]["union"]["supported_views"],
        "UnionMatched": f'{contrast["point"]:+.4f}',
        "UnionCI": "["+", ".join(f"{k:+.4f}" for k in contrast["interval95"])+"]",
    }
    return values, {str(f.relative_to(home)):BASE.cal.digest(f) for f in sources}


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        marker = "Neither oracle is deployable."
        addition = (" A broader learned CV--OLS4 policy then regained coverage but "
                    "did not beat OLS4. Repairing predicted feasibility yielded fewer "
                    "actual passes than a matched past-hindcast control (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), marker, marker+addition))
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC/"supplement_dense_motion.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+"\n".join(
            rf"\newcommand{{\Dense{k}}}{{{v}}}" for k,v in values.items())+"\n")
        refs[str(frag.relative_to(PUBLIC.parent))] = BASE.cal.digest(frag)
        files = [tex, defs]
    doc["source_bindings"].update(refs)
    for f in files:
        doc[key][f.name] = BASE.cal.digest(f)
    (out/manifest).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("main", "supplement"))
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--source-zip", type=Path)
    cli.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    cli.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = cli.parse_args()
    out = a.output_dir or DEFAULT/("build" if a.part=="main" else "supplement")
    result = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            previous.previous.prior.main.v22.main.v13.package(out, a.source_zip, result["build_bindings"])
        else:
            BASE.v22.supp.prior.package(out, a.source_zip, result)
