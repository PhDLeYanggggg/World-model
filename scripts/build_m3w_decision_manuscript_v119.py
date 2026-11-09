"""Bind the DUT matched-motion readout to the main text and supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v116 as main_parent
from scripts import build_m3w_decision_manuscript_v118 as supp_parent
from scripts import build_m3w_dut_matched_replay as dut

BASE, PACK = supp_parent.BASE, supp_parent.PACK
PUBLIC = supp_parent.PUBLIC.parent / "evidence_manuscript_v119"
DEFAULT = supp_parent.DEFAULT.parent / PUBLIC.name


def number(x):
    return f"{x:+.2f}"


def build(out, style, archive, part):
    result = dut.inputs()
    scalars, expected = dut.collect()
    dut.replay.verify(dut.replay.summarize(scalars), expected)
    p = expected["paired"]
    damp, ols = "damped_velocity_005", "ols4"
    bindings = [dut.OUT / f"{k}.json" for k in ("analysis", "registration", "full_ols4")]
    full = json.loads(bindings[-1].read_text())
    registration = json.loads(bindings[1].read_text())
    if (full["original"] != registration["prior"] or full["confirmation_access"]
            or full["new_neural_inference"] or not full["complete_case_only"]):
        raise ValueError("Require the full-cohort OLS4 check with frozen neural errors")
    counts = result["counts"]
    macros = {k: f"{v:,}" for k, v in {
        "Queries": counts["queries"], "Targets": counts["targets"], "Complete": counts["complete"],
        "StrictCount": expected["strict"], "MatchedCount": expected["common"],
        "FullTargets": full["totals"]["targets"], "FullComplete": full["totals"]["complete"]}.items()}
    sites = list(full["sites"].values())
    for metric, label in (("ade", "ADE"), ("fde", "FDE")):
        m = [s["ols4"][metric] for s in sites]
        e = [v[metric] for s in sites for v in s["uncontrolled"].values()]
        macros[f"FullOLS{label}"] = f"{sum(m) / len(m):.3f}"
        macros[f"FullEq{label}"] = f"{sum(e) / len(e):.3f}"
    wins = sum(s["ols4"]["ade"] < v["ade"] for s in sites for v in s["uncontrolled"].values())
    macros["FullWins"] = str(wins)
    for key, name in ((f"raw_vs_{damp}", "RawDamp"), (f"raw_vs_{ols}", "RawOLS"),
                      (f"strict_vs_{damp}", "StrictDamp"), (f"strict_vs_{ols}", "StrictOLS"),
                      (f"matched_vs_{damp}", "MatchedDamp"), (f"matched_vs_{ols}", "MatchedOLS")):
        macros[name] = " & ".join(number(p[key][f]["point"])
                                  for f in ("complete_gain_pct", "lower_pp", "upper_pp"))
    macros["MatchedOLSUpper"] = number(p[f"matched_vs_{ols}"]["upper_pp"]["point"])
    lo, hi = p[f"matched_vs_{damp}"]["lower_pp"]["interval95"]
    macros["MatchedInterval"] = f"[{number(lo)}, {number(hi)}]"
    for k, name in (("eqmotion_raw", "RawEasy"), ("strict", "StrictEasy"),
                    ("matched_neural", "MatchedEasy"), (f"matched_{damp}", "DampEasy"),
                    (f"matched_{ols}", "OLSEasy")):
        macros[name] = number(expected["arms"][k]["worst_easy_pct"])
    commands = "\n".join(rf"\newcommand{{\DutMotion{k}}}{{{v}}}" for k, v in macros.items()) + "\n"
    if part == "main":
        doc = main_parent.build(out, style, archive, part)
        tex, defs = out / "manuscript.tex", out / "evidence_values.tex"
        text = ("On TUMDOT development, aligned peer context does not beat blind or permuted "
                "controls at matched query counts (supplement).")
        update = (r"On exposed DUT, OLS4 beats raw EqMotion in \DutMotionFullWins{}/6 site--seed "
                  r"views. On disjoint queries, even the neural upper gain at matched recipients "
                  r"and peak displacement is \DutMotionMatchedOLSUpper{} points versus OLS4 (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), text, update))
        receipt, manifest_key = "source_bindings.json", "build_bindings"
    else:
        doc = supp_parent.build(out, style, archive)
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        frag = PUBLIC / "dut_matched_motion.tex"
        bindings.append(frag)
        text = r"\section*{LLM Usage Statement}"
        content = BASE.replace_once(tex.read_text(), text, frag.read_text() + "\n" + text)
        bib = (r"\bibitem{dut} D. Yang, L. Li, K. Redmill, and U. Ozguner. Top-view trajectories: "
               r"A pedestrian dataset of vehicle-crowd interaction from controlled experiments and "
               r"crowded campus. In \emph{IEEE Intelligent Vehicles Symposium}, 2019.")
        tex.write_text(BASE.replace_once(content, r"\end{thebibliography}", bib + "\n\n" + r"\end{thebibliography}"))
        receipt, manifest_key = "supplement_bindings.json", "files"
    defs.write_text(defs.read_text() + "\n" + commands)
    doc["source_bindings"].update({str(f.relative_to(PUBLIC.parent)): BASE.cal.digest(f) for f in bindings})
    doc[manifest_key].update({f.name: BASE.cal.digest(f) for f in (tex, defs)})
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
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
    result = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == "main":
        PACK.main_package.package(out, a.source_zip, result["build_bindings"])
    elif a.source_zip:
        PACK.supplement_package.package(out, a.source_zip, result)
