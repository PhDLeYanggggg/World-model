"""Bind the reference-loss guard and its exact proposal thinning audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v103 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v104"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    guard = timing.legacy.read(timing.REFERENCE_LOSS)["summary"]
    audit = timing.read_reference_loss_oracle()["summary"]
    sha = BASE.cal.digest
    paths = [PUBLIC.parent/timing.REFERENCE_LOSS/f"{f}.json" for f in
             ("registration", "analysis", "head_freeze", "oracle_registration", "oracle_analysis")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = ("Gain deferral loses to immediate use at matched budgets. Earlier support "
               "increases interventions but can lose easy references (supplement).")
        new = ("Gain deferral loses to immediate use at matched budgets. A separate reference-loss "
               "guard cannot recover the original joint pass count (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"reference_loss.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    d, a = guard["diagnostics"], audit["diagnostics"]
    vals = dict(Proposed=d["proposals"], Accepted=d["direct_accepted"],
                Before=d["frozen_lost_reference"], After=d["direct_lost_reference"],
                Shortfall=a["no_loss_quota_shortfall"])
    for s, k, tag in ((guard, "original", "Original"), (guard, "frozen", "Frozen"),
                      (guard, "direct", "Direct"), (guard, "factorized", "Factored"),
                      (guard, "direct_easy_matched", "Matched"), (audit, "exact_joint", "Exact"),
                      (audit, "no_loss_at_most", "NoLoss")):
        m = s["metrics"][k]
        vals[tag+"Counts"] = f'{m["edits"]} / {m["dual_positive"]} / {m["undefined"]}'
    for k, tag in (("direct_vs_frozen", "Frozen"), ("direct_vs_direct_easy_matched", "Matched")):
        e = guard["effects"][k]
        lo, hi = e["interval95"]
        vals[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(
        rf"\newcommand{{\RefLoss{k}}}{{{x}}}" for k, x in vals.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in paths})
    doc[bindings].update({f.name: sha(f) for f in (tex, defs)})
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
        packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        packing.supplement_package.package(out, a.source_zip, doc)
