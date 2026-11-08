"""Include the frozen-calibration intervention-capacity result in the paper."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v36 as previous

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v37"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE
SOURCE = PUBLIC.parent/"dronecrowd_perfect_switch_ceiling_v1"


def evidence():
    p, reg = SOURCE/"analysis.json", SOURCE/"registration.json"
    score, replay = [SOURCE/f"{mode}_execution.json" for mode in ("score", "replay")]
    d = json.loads(p.read_text())
    sha = BASE.cal.digest
    receipts = [json.loads(f.read_text()) for f in (score, replay)]
    if (any(r["analysis_sha256"] != sha(p) for r in receipts)
            or receipts[1]["calibration_labels_reopened"]
            or d["registration_sha256"] != sha(reg) or not d["original_ols4_readout_reproduced"]
            or d["confirmation_access"] or d["risk_rule_changed"] or d["views"] != 120
            or d["unrestricted_witness_supported_views"] != 120 or d["original_joint_switches"] != 67440
            or not all(z["capped_perfect_mass"] < z["ols4_lower_pp"] for z in d["groups"].values())):
        raise ValueError("Require replayed, calibration-only capacity and witness evidence")
    metrics = d["metrics"]
    rows = []
    for key, title in (
        ("capped_zero_harm_menu_gain_mass", "Capped menu vs. CV"),
        ("capped_perfect_mass", "Capped upper vs. CV"),
        ("ols4_lower_pp", "OLS4 lower vs. CV"),
        ("capped_perfect_upper_vs_ols4_mass", "Capped upper vs. OLS4"),
        ("unrestricted_zero_harm_menu_gain_mass", "Uncapped witness vs. CV"),
    ):
        m = metrics[key]
        lo, hi = m["ci95"]
        rows.append(f'{title} & {m["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}] '+r'\\')
    refs = {str(f.relative_to(PUBLIC.parent)):sha(f) for f in (p, reg, score, replay)}
    return metrics, "\n".join(rows), refs


def build(out, style, archive, part):
    m, rows, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        before = r"\section{Separating Movement, Support, and Decision Quality}"
        paragraph = (
            f'At the original per-query cap, even a zero-error oracle on the '
            f'best recipients could gain only {m["capped_perfect_mass"]["point"]:+.2f} points '
            f'against CV, versus {m["ols4_lower_pp"]["point"]:+.2f} for unrestricted OLS4. '
            r'An unrestricted outcome-aware menu witness passes all 120 views, '
            r'separating action coverage from the ability to select safely before labels '
            r'(supplement). Neither oracle is deployable.'
        )
        text = BASE.replace_once(tex.read_text(), before, paragraph+"\n\n"+before)
        tex.write_text(text)
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        fragment = PUBLIC/"supplement_perfect_ceiling.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, fragment.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+r"\newcommand{\CeilingRows}{"+rows+"}\n")
        refs[str(fragment.relative_to(PUBLIC.parent))] = BASE.cal.digest(fragment)
        files = [tex, defs]
    doc["source_bindings"].update(refs)
    for f in files:
        doc[key][f.name] = BASE.cal.digest(f)
    (out/manifest).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
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
    d = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            previous.prior.main.v22.main.v13.package(out,a.source_zip,d["build_bindings"])
        else:
            BASE.v22.supp.prior.package(out,a.source_zip,d)
