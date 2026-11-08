"""Report the frozen neural bank's exact timing capacity in the main paper."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v85 as previous
from scripts import build_m3w_decision_manuscript_v81 as main_base

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v86"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_release_credit_v1"
    files = [root/f"{k}.json" for k in ("registration", "analysis")]
    reg, a = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if a["registration"] != sha(files[0]) or any(v["confirmation_access"] or not v["original_2pct"] for v in (reg, a)):
        raise ValueError("Require unchanged risk and sealed confirmation")
    s = a["summary"]
    cap = s["capacity"]
    metrics = s["metrics"]
    views = metrics["immediate"]["views"]
    if any(sum(v.values()) != views for v in cap.values()) or any(v.get("unresolved", 0) for v in cap.values()):
        raise ValueError("Require completely adjudicated timing capacity")

    def effect(k):
        v = s["effects"][f"{k}_vs_matched"]
        lo, hi = v["interval95"]
        return f'{v["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    val = dict(Views=views, Fixed=cap["0"]["qualified"], Short=cap["2"]["qualified"], Long=cap["8"]["qualified"],
               FixedNoEasy=cap["0"]["no_eligible_easy"], ShortNoEasy=cap["2"]["no_eligible_easy"],
               LongNoEasy=cap["8"]["no_eligible_easy"], NoRelease=cap["0"]["no_edits"],
               Edits=metrics["immediate"]["edits"], FixedEffect=effect("fixed_wait"),
               ValueEffect=effect("value_wait"), SupportEffect=effect("support_wait"))
    for k, label in dict(immediate="Now", fixed_wait="Delay", value_wait="Value", support_wait="Support").items():
        for name, field in dict(Edits="edits", Risk="risk_pass", Joint="dual_positive").items():
            val[label+name] = metrics[k][field]
        if k != "immediate":
            match = metrics[f"{k}_matched"]
            if match["edits"] != metrics[k]["edits"]:
                raise ValueError("The timing comparison must use the same realized count")
            val[label+"Expired"] = s["schedule"][k]["expired"]
            val[label+"MatchedJoint"] = match["dual_positive"]
    if part == "main":
        doc = main_base.build(out, style, archive, part)
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"One-edit event supervision does not beat frozen moments. "
               r"One-edit capacity rises from \FirstEarly{} at the first query to "
               r"\FirstAny{} at any query, yet donor-selected stopping loses "
               r"\StopEffect{} to immediate on \StopMatched{} count-matched "
               r"views (supplement).")
        new = (r"With the frozen neural bank and identical recording totals, delaying "
               r"released edits by up to two queries raises hindsight capacity from "
               r"\NeuralCreditFixed{} to \NeuralCreditShort{} of \NeuralCreditViews{} views. "
               r"The causal value rule yields \NeuralCreditValueEffect{} against "
               r"immediate spending of the same realized credits; the "
               r"support-priority rule yields \NeuralCreditSupportEffect{} (supplement).")
        result = BASE.replace_once(tex.read_text(), old, new)
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        doc = previous.build(out, style, archive)
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"release.tex"
        marker = r"\section*{LLM Usage Statement}"
        result = BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker)
        files.append(frag)
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(result)
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\NeuralCredit{k}}}{{{v}}}" for k, v in val.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in files})
    doc[bind].update({p.name: sha(p) for p in (tex, defs)})
    (out/receipt).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("main", "supplement"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = a.output_dir or DEFAULT/("build" if a.part == "main" else "supplement")
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == "main":
        previous.packaging.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        previous.packaging.supplement_package.package(out, a.source_zip, doc)
