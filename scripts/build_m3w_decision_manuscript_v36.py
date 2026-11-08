"""Add the seed-shared missing-label result to the anonymized draft."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v35 as prior

PUBLIC = prior.PUBLIC.parent/"evidence_manuscript_v36"
DEFAULT = prior.DEFAULT.parent/PUBLIC.name
BASE = prior.base
SOURCE = PUBLIC.parent/"opendd_seed_shared_completion_v1"


def evidence():
    reg, analysis = SOURCE/"registration.json", SOURCE/"analysis.json"
    data = json.loads(analysis.read_text())
    if (data["registration_sha256"] != BASE.cal.digest(reg) or data["fit_count"] != 0
            or not data["original_per_seed_lower_scores_reproduced"]
            or not data["original_risk_unchanged"] or data["independent_confirmation"]):
        raise ValueError("Require the verified frozen shared-seed analysis")
    c = data["contrasts"]
    if not all(c["traffic_vs_"+k]["witness_sign_reversal"] for k in ("pedestrian", "shuffled_traffic")):
        raise ValueError("The manuscript statement requires both direct sign reversals")
    if c["traffic_vs_half"]["shared_lower"]["point"] <= 0:
        raise ValueError("The manuscript statement requires the fixed-half contrast")
    rows = []
    for k, name in (("half", "Fixed half"), ("pedestrian", "Pedestrian context"),
                    ("shuffled_traffic", "Shuffled motor")):
        x = c["traffic_vs_"+k]
        fmt = lambda m: f'{x[m]["point"]:+.4f}'
        rows.append(f'{name} & {fmt("observed")} & '
                    f'[{fmt("shared_lower")}, {fmt("shared_upper")}] & '
                    f'[{fmt("template_low")}, {fmt("template_high")}] ' + r'\\')
    refs = {str(p.relative_to(PUBLIC.parent)):BASE.cal.digest(p) for p in (reg, analysis)}
    return "\n".join(rows), c["traffic_vs_half"]["missing_steps"], refs


def build(out, style, archive, part):
    rows, n, refs = evidence()
    doc = prior.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        old = "All OpenDD results remain development evidence."
        new = ("With one missing label shared across training seeds, explicit "
               "trajectory completions still reverse both direct context "
               "comparisons (supplement). "+old)
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        fragment = PUBLIC/"supplement_seed_completion.tex"
        tex.write_text(BASE.replace_once(tex.read_text(), r"\section*{LLM Usage Statement}",
                                         fragment.read_text()+"\n\n"+r"\section*{LLM Usage Statement}"))
        defs.write_text(defs.read_text()+r"\newcommand{\SeedCompletionRows}{"+rows+"}\n"+
                        rf"\newcommand{{\SeedCompletionMissing}}{{{n:,}}}"+"\n")
        refs[str(fragment.relative_to(PUBLIC.parent))] = BASE.cal.digest(fragment)
        files = [tex, defs]
    doc["source_bindings"].update(refs)
    for p in files:
        doc[key][p.name] = BASE.cal.digest(p)
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
    result = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            prior.main.v22.main.v13.package(out, a.source_zip, result["build_bindings"])
        else:
            BASE.v22.supp.prior.package(out, a.source_zip, result)
