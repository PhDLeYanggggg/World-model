"""Add paired sampling sensitivity and the convex risk-capacity certificate."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v40 as previous

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v41"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    files = [home/"past_sampling_audit_v1"/k for k in
             ("analysis.json", "fitting_capacity.json", "convex_capacity.json")]
    r, discrete, convex = [json.loads(f.read_text()) for f in files]
    if (r["confirmation_opened"] or r["risk_rule_changed"] or discrete["independent_confirmation"]
            or convex["confirmation_access"] or discrete["parent"]["sha256"] != BASE.cal.digest(files[0])
            or convex["parent"]["sha256"] != BASE.cal.digest(files[1])):
        raise ValueError("Need bound verified sampling and fitting capacity only")
    rows = []
    for arm, name in (("recent", "Recent four"), ("wide_uniform", "Wide uniform four"),
                      ("wide_endpoints", "Wide endpoint four"), ("wide_dense", "Wide dense seven")):
        effect = [r[k]["effects"].get(arm+"_vs_recent", {}).get("completion_lower_pp", {}).get("point", 0.)
                  for k in ("fitting", "opened_calibration")]
        fit, opened = [r[k]["risk"][arm]["supported_views"] for k in ("fitting", "opened_calibration")]
        rows.append(f"{name} & {effect[0]:+.2f} & {effect[1]:+.2f} & {fit}/{opened} "+r'\\')
    v = dict(Rows="\n".join(rows), Blocked=convex["excluded_by_optimistic_bound"], Fit=convex["recordings"])
    for k, label in (("fitting", "Source"), ("opened_calibration", "Target")):
        e = r[k]["effects"]["wide_uniform_vs_recent"]["completion_lower_pp"]
        v[label] = "["+", ".join(f"{x:+.3f}" for x in e["interval95"])+"]"
    return v, {str(f.relative_to(home)): BASE.cal.digest(f) for f in files}


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        old = "the reversal (supplement)."
        new = ("the reversal. Wider past sampling improved the opened-cohort "
               "error but hurt fitting sources. Even stepwise convex mixtures "
               "of these schedules could not repair fitting risk at fixed "
               "recipients (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC/"supplement_past_sampling.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+"\n".join(
            rf"\newcommand{{\PastSample{k}}}{{{v}}}" for k, v in values.items())+"\n")
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
            previous.previous.previous.previous.previous.prior.main.v22.main.v13.package(out, a.source_zip, d["build_bindings"])
        else:
            BASE.v22.supp.prior.package(out, a.source_zip, d)
