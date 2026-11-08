"""Integrate the cross-window result into the main paper and supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v13 as main_package
from scripts import build_m3w_decision_manuscript_v73 as main_previous
from scripts import build_m3w_decision_supplement_v20 as supplement_package
from scripts import build_m3w_decision_supplement_v78 as supplement_previous

BASE = main_previous.BASE
PUBLIC = main_previous.PUBLIC.parent / "evidence_manuscript_v79"
DEFAULT = main_previous.DEFAULT.parent / PUBLIC.name
OLD = (
    r"At the same edit counts and radii, permitting two-query delay raises fixed-time "
    r"witnesses from \TimingFixed{} to \TimingShort{}. Neither greedy nor learned deferral "
    r"recovers the gain. The actual decoder can reach \LedgerToggle{} versus \LedgerAnchor{} "
    r"with a label-aware, count-preserving single change per recording. In \TimingQuota{} views, "
    r"the streaming budget cannot fund even one edit."
)
NEW = (
    r"Label-aware delay expands capacity from \TimingFixed{} to \TimingShort{}, without "
    r"a causal gain. With the first path fixed, moving the second edit later gives "
    r"\CrossTiming{} on \CrossCommon{} completed pairs. At identical cross-query clocks, "
    r"joint scoring gives \CrossJoint{} over independent scoring and passes "
    r"\CrossJointPass{} versus \CrossIndependentPass{} joint checks."
)


def build(out, style, archive, part):
    root = PUBLIC.parent / "cross_query_edit_v2"
    paths = [root / f"{k}.json" for k in ("registration", "analysis")]
    r, a = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if a["registration"] != sha(paths[0]) or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a)):
        raise ValueError("Require the bound causal timing and coordination comparison")
    s = a["summary"]

    def effect(k, decimals):
        v = s[k]
        lo, hi = v["interval95"]
        return f'{v["point"]:+.{decimals}f} pp [{lo:+.{decimals}f}, {hi:+.{decimals}f}]'

    count, metric, d = s["count"], s["metrics"], s["diagnostic"]
    j, i = (metric[k] for k in ("early_joint", "early_independent"))
    if j["edits"] != i["edits"]:
        raise ValueError("Coordination comparison must match edits")
    values = dict(Timing=effect("timing_pp", 3), Joint=effect("joint_pp", 4), Common=d["timing_common"],
        Edits=f'{j["edits"]:,}', JointPass=j["dual_positive"], IndependentPass=i["dual_positive"],
        SamePass=metric["same"]["dual_positive"], SplitPass=metric["split_same_start"]["dual_positive"],
        Incomplete=count["split_same_start"]["1"], Next=d["first_fixed_next_query_witness"],
        Any=d["first_fixed_any_later_witness"])
    if part == "main":
        doc = main_previous.build(out, style, archive, part)
        tex, defs = out / "manuscript.tex", out / "direct_count_values.tex"
        content = BASE.replace_once(tex.read_text(), OLD, NEW)
        bindings, receipt = "build_bindings", "source_bindings.json"
    else:
        doc = supplement_previous.build(out, style, archive)
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        frag = PUBLIC / "cross_query.tex"
        marker = r"\section*{LLM Usage Statement}"
        content = BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker)
        paths.append(frag)
        bindings, receipt = "files", "supplement_bindings.json"
    tex.write_text(content)
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Cross{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in paths})
    doc[bindings].update({p.name: sha(p) for p in (tex, defs)})
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
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
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == "main":
        main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        supplement_package.package(out, a.source_zip, doc)
