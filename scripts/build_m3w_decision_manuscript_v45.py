"""Bind the nested source-disjoint calibration study to the working paper."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v44 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v45"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    experiment = home / "nested_recipient_calibration_v1"
    files = [experiment / f"{name}.json" for name in ("analysis", "freeze", "frontier", "support_audit")]
    refs = {str(p.relative_to(home)): BASE.cal.digest(p) for p in files}
    a, freeze, frontier, support = [json.loads(p.read_text()) for p in files]
    if (frontier["analysis"]["sha256"] != BASE.cal.digest(files[0])
            or support["frontier"]["sha256"] != BASE.cal.digest(files[2])
            or not freeze["all_outer_choices_frozen_before_score"]
            or not a["no_official_calibration"] or freeze["official_calibration_access"]
            or any(d["confirmation_access"] for d in (a, freeze, frontier, support))
            or frontier["thresholds_changed"] or support["thresholds_changed"]
            or support["outer_evaluation_used"] or not support["hindsight_risk_only"]
            or a["fits"] != 27 or any(a["selected_folds"].values())
            or any(row["folds_with_any_full_risk"] for row in frontier["summary"].values())):
        raise ValueError("Require frozen nested development calibration and its bound diagnostics")
    risk = a["summary"]["risk"]
    if risk["fixed_gain"]["switches"] != risk["fixed_support"]["switches"]:
        raise ValueError("Fixed-quota rankers must have the same number of interventions")
    effect = a["summary"]["effects"]["fixed_support_vs_fixed_gain"]["completion_lower_pp"]
    v = {
        "Fits": a["fits"], "Views": a["summary"]["views"], "Actions": risk["fixed_gain"]["switches"],
        "Effect": f'{effect["point"]:+.3f}',
        "Interval": "[" + ", ".join(f"{x:+.3f}" for x in effect["interval95"]) + "]",
        "Levels": sum(row["evaluated_levels"] for row in frontier["summary"].values()),
        "Unsupported": support["folds_with_missing_easy_roster"],
        "Supported": support["source_folds"] - support["folds_with_any_unsupported_recording"]["nominee"],
    }
    return v, refs


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out / "manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        old = "Query-local fallback makes few changes and shows no utility lift (supplement)."
        new = "Query-local fallback and source-disjoint calibration yield no utility lift (supplement)."
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        touched = [tex]
    else:
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC / "supplement_nested_recipient.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text() + "\n\n" + marker))
        defs.write_text(defs.read_text() + "\n".join(
            rf"\newcommand{{\NestedRecipient{k}}}{{{value}}}" for k, value in values.items()) + "\n")
        refs[str(frag.relative_to(PUBLIC.parent))] = BASE.cal.digest(frag)
        touched = [tex, defs]
    doc["source_bindings"].update(refs)
    for p in touched:
        doc[key][p.name] = BASE.cal.digest(p)
    (out / manifest).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("main", "supplement"))
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--source-zip", type=Path)
    cli.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    cli.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    args = cli.parse_args()
    out = args.output_dir or DEFAULT / ("build" if args.part == "main" else "supplement")
    doc = build(out, args.style, args.archive, args.part)
    if args.source_zip:
        if args.part == "main":
            previous.previous.main_package.package(out, args.source_zip, doc["build_bindings"])
        else:
            previous.previous.supp_package.package(out, args.source_zip, doc)
