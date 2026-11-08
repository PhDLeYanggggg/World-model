"""Bind the matched OpenDD completion-training experiment into the supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v117 as previous
from scripts import build_m3w_opendd_completion_replay as replay

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v118"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    packet, expected = replay.collect()
    replay.base.replay.check(replay.base.replay.recompute(packet), expected)
    doc = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "completion_forecast.tex"
    anchor = r"\section*{LLM Usage Statement}"
    body = BASE.replace_once(tex.read_text(), anchor, frag.read_text() + "\n" + anchor)
    end = r"\end{thebibliography}"
    bib = (r"\bibitem{trajimpute} P. S. Chib and P. Singh. Pedestrian trajectory prediction with missing "
           r"data: Datasets, imputation, and benchmarking. In \emph{NeurIPS Datasets and Benchmarks}, 2024.")
    tex.write_text(BASE.replace_once(body, end, bib + "\n\n" + end))
    primary, radius = expected["completion_objectives"], expected["realized_peak_control"]

    def f(e):
        lo, hi = e["interval95"]
        return f'{e["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    macros = {}
    for k, label in (("masked", "Masked"), ("cv_completion", "CV"), ("dual_completion", "Dual")):
        c = primary["counts"][f"matched_{k}"]
        macros[label] = " & ".join((f(primary["effects"][f"raw_{k}_cv"]),
            str(primary["counts"][f"raw_{k}"]["risk_pass"]), str(c["dual_positive"])))
        if k != "masked":
            macros[f"{label}Direct"] = f(primary["effects"][f"direct_{k}_lower"])
            macros[f"{label}Radius"] = f(radius["effects"][f"{k}_lower"])
    defs.write_text(defs.read_text() + "\n" + "\n".join(
        rf"\newcommand{{\Completion{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    report = PUBLIC.parent / replay.NAME
    paths = [report / f"{k}.json" for k in ("analysis", "registration", "freeze", "radius_analysis")]
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): BASE.cal.digest(p) for p in (*paths, frag)})
    doc["files"].update({p.name: BASE.cal.digest(p) for p in (tex, defs)})
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    args = p.parse_args()
    result = build(args.output_dir, args.style, args.archive)
    if args.source_zip:
        PACK.supplement_package.package(args.output_dir, args.source_zip, result)
