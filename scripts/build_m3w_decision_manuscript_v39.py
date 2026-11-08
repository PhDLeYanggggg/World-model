"""Add the fitted standalone hindcast and matched-recipient risk comparison."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v38 as previous

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v39"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    paths = [home/"standalone_hindcast_v1"/k for k in ("fitting.json", "analysis.json", "risk_gap.json")]
    fit, result, audit = [json.loads(p.read_text()) for p in paths]
    if (fit["registration"] != result["registration"] or fit["chosen"] != result["chosen"]
            or result["calibration_outcomes_used_to_select"] or result["independent_confirmation"]
            or result["risk_rule_changed"] or audit["analysis"]["sha256"] != BASE.cal.digest(paths[1])
            or result["summary"]["views"] != 40 or result["matched_queries"] != 11240):
        raise ValueError("Require a fitting-only choice and exact opened-calibration results")
    gain, risk = result["summary"]["effects"], result["summary"]["risk"]
    rows = []
    for arm, name in (("history_ols4_velocity", "OLS4"), ("fixed_half", "Fixed half"),
                      ("standalone", "Three-origin gate"), ("matched_latest_origin", "Matched latest origin")):
        rows.append(f'{name} & {gain[arm+"_vs_cv"]["completion_lower_pp"]["point"]:+.2f} & '
                    f'{risk[arm]["supported_views"]} '+r'\\')
    contrast = gain["standalone_vs_matched_latest_origin"]["completion_lower_pp"]
    values = {
        "Rows": "\n".join(rows),
        "Matched": f'{contrast["point"]:+.3f}',
        "MatchedCI": "["+", ".join(f"{v:+.3f}" for v in contrast["interval95"])+"]",
        "Violation": audit["fitting"]["observed_only_violations"],
        "FitCount": audit["fitting"]["recordings"],
        "OOFPass": sum(v["passed"] for v in fit["left_out"].values()),
    }
    return values, {str(p.relative_to(home)): BASE.cal.digest(p) for p in paths}


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        before = "yielded fewer actual passes than a matched past-hindcast control (supplement)."
        add = ("yielded fewer actual passes than a matched past-hindcast control. "
               "A fitting-selected hindcast alone passed all 40 opened clips but failed "
               "on its fitting domain (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), before, add))
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC/"supplement_standalone_hindcast.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+"\n".join(
            rf"\newcommand{{\Hind{k}}}{{{v}}}" for k, v in values.items())+"\n")
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
    out = a.output_dir or DEFAULT/("build" if a.part == "main" else "supplement")
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            previous.previous.previous.prior.main.v22.main.v13.package(out, a.source_zip, doc["build_bindings"])
        else:
            BASE.v22.supp.prior.package(out, a.source_zip, doc)
