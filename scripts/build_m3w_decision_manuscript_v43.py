"""Add a three-seed matched-budget risk-head and recipient diagnostic."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v42 as previous
from scripts import build_m3w_decision_manuscript_v13 as main_package
from scripts import build_m3w_decision_supplement_v20 as supp_package

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v43"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    folder = home/"matched_support_moments_v1"
    paths = [folder/k for k in ("analysis.json", "mass_audit.json", "recipient_audit.json", "freeze.json")]
    result, mass, recipient, freeze = [json.loads(p.read_text()) for p in paths]
    if (mass["analysis"]["sha256"] != BASE.cal.digest(paths[0])
            or recipient["parent"]["sha256"] != BASE.cal.digest(paths[3]) or result["confirmation_access"]
            or freeze["confirmation_access"] or not freeze["all_held_choices_frozen_before_scoring"]
            or result["fitted_models"] != 54 or result["matched_switches_per_arm"] != 95769):
        raise ValueError("Require the verified matched held-source experiment")
    summary = result["summary"]
    rows = []
    labels = (("marginal_gain", "Marginal gain"), ("marginal_record", "Marginal record"),
              ("supervised_gain", "Conditional gain"), ("supervised_record", "Conditional record"),
              ("supervised_query", "Conditional query"))
    for arm, label in labels:
        risk = summary["risk"][arm]
        val = summary["effects"][arm+"_vs_cv"]["completion_lower_pp"]["point"]
        rows.append(f'{label} & {val:+.2f} & {risk["all_risk_over_2_views"]} & '
                    f'{risk["supported_views"]} '+r'\\')
    effect = summary["effects"]["supervised_record_vs_marginal_record"]["completion_lower_pp"]
    values = dict(Rows="\n".join(rows), Direct=f'{effect["point"]:+.3f}',
        Interval="["+", ".join(f"{v:+.3f}" for v in effect["interval95"])+"]",
        MarginalOver=mass["summary"]["marginal_record"]["reference_overestimates"][1],
        ConditionalOver=mass["summary"]["supervised_record"]["reference_overestimates"][1],
        FixedRecipient=recipient["summary"]["supervised_record"]["supported"])
    return values, {str(p.relative_to(home)): BASE.cal.digest(p) for p in paths}


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        old = next(line for line in tex.read_text().splitlines()
                   if line.startswith("At the original per-query cap,"))
        new = ("On the opened cohort, unrestricted OLS4 gains +51.29 points over CV, "
               "versus a perfect-switch upper bound of +5.29 at the original query cap. "
               "Broader candidate menus provide hindsight capacity but do not establish a "
               "transferable policy. In the fitting cohort, redistributing fixed recording "
               "totals yields 14/24 witnesses against CV and OLS4. At those totals, three-seed "
               "neural risk heads fail the gate; hindsight path swaps cannot rescue their "
               "selected recipients (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC/"supplement_matched_support.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+"\n".join(
            rf"\newcommand{{\MatchSupport{k}}}{{{v}}}" for k, v in values.items())+"\n")
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
    d = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            main_package.package(out, a.source_zip, d["build_bindings"])
        else:
            supp_package.package(out, a.source_zip, d)
