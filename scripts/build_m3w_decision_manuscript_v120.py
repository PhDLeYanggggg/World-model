"""Bind the expert-mixture ablation and lead diagnostic to the supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v119 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / "evidence_manuscript_v120"
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    report = PUBLIC.parent / "moving_prefix_expert_mixture_v1"
    readout, leads, reg = (report / f"{n}.json" for n in ("readout", "lead_alignment", "registration"))
    a, b = (json.loads(p.read_text()) for p in (readout, leads))
    if (a["registration"]["sha256"] != BASE.cal.digest(reg)
            or a["raw_local"] != b["parent"] or a["confirmation_access"] or b["confirmation_access"]
            or not a["original_2pct"]):
        raise ValueError("Require the bound prefix-expert evidence")
    doc = parent.build(out, style, archive, "supplement")
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "prefix_expert.tex"
    anchor = r"\section*{LLM Usage Statement}"
    body = BASE.replace_once(tex.read_text(), anchor, frag.read_text() + "\n" + anchor)
    end = r"\end{thebibliography}"
    bib = (r"\bibitem{fforma} P. Montero-Manso, G. Athanasopoulos, R. J. Hyndman, "
           r"and T. S. Talagala. FFORMA: Feature-based forecast model averaging. "
           r"\emph{International Journal of Forecasting}, 36(1):86--92, 2020.")
    tex.write_text(BASE.replace_once(body, end, bib + "\n\n" + end))
    e, m = a["raw"]["effects"], a["matched"]["effects"]
    macros = {"Count": f'{a["matched_edits"]:,}', "Complete": f'{b["complete_rows"]:,}'}
    for k, r in (("RawKin", e["prefix_vs_kinematic"]), ("RawOLS", e["prefix_vs_history_ols4_velocity"]),
                 ("MatchedKin", m["kinematic"]), ("MatchedOLS", m["ols4"])):
        macros[k] = f'{r["known_pp"]["point"]:+.3f} & {r["completion_lower_pp"]["point"]:+.3f}'
    for lead, label in (("1", "One"), ("4", "Four"), ("8", "Eight"), ("12", "Twelve")):
        r = b["leads"][lead]["summary"]
        macros[label] = f'{100 * r["agreement"]["point"]:.1f} & {r["prefix_winner_vs_ols4_pp"]["point"]:+.3f}'
    defs.write_text(defs.read_text() + "\n" + "\n".join(
        rf"\newcommand{{\PrefixExpert{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): BASE.cal.digest(p)
                                   for p in (readout, leads, reg, frag)})
    doc["files"].update({p.name: BASE.cal.digest(p) for p in (tex, defs)})
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        PACK.supplement_package.package(a.output_dir, a.source_zip, doc)
