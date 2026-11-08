"""Bind pedestrian sparse-support comparisons and strong-baseline capacity."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v106 as previous
from scripts import build_m3w_sparse_support_replay as sparse

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v107"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a, b, c = sparse.inputs()
    sha = BASE.cal.digest
    paths = [sparse.SOURCE/f"{f}.json" for f in ("registration", "analysis", "head_freeze",
             "capacity_registration", "capacity", "bridge_registration", "bridge")]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = ("Broader support directions increase hindsight capacity, but direct pass-transition "
               "supervision still loses original passes (supplement).")
        new = ("On pedestrian development, one-edit support is often feasible against CV, "
               "but rarely beats both motion baselines (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"sparse_support.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    s, b, c = a["summary"], b["summary"], c["summary"]
    vals = dict(Strong=b["ols_certified_better"], EasyAll=b["safe_class_unlimited"],
                BridgeEight=c["bridge_8"], BridgeAll=c["bridge_all"],
                StrongBridge=c.get("strong_bridge_all", 0))
    for k, tag in (("gain", "Gain"), ("cv_head", "CVHead"), ("dual_head", "DualHead"),
                   ("first_cv_oracle", "FirstCV"), ("first_dual_oracle", "FirstDual"),
                   ("window_cv_oracle", "WindowCV"), ("window_dual_oracle", "WindowDual")):
        m = s["metrics"][k]
        vals[tag+"Counts"] = f'{m["edits"]} / {m["cv_joint"]} / {m["dual_positive"]}'
    for k, tag in (("cv_head_vs_gain", "Matched"), ("cv_head_ols", "Strong")):
        e = s["effects"][k]
        lo, hi = e["interval95"]
        vals[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(
        rf"\newcommand{{\SparseSupport{k}}}{{{x}}}" for k, x in vals.items())+"\n")
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
