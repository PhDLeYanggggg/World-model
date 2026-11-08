"""Bind the direct recipient-ranking and site-sensitivity results."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v88 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v89"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_slate_rank_v1"
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "head_freeze", "sensitivity_registration", "sensitivity")]
    r, a, h, sr, s = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or h["registration"] != a["registration"] or a["heads"] != sha(paths[2])
            or s["registration"] != sha(paths[3]) or sr["parent"]["sha256"] != sha(paths[1])
            or not r["all_arms_match_query_counts"] or not sr["primary_unchanged"]
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a, sr, s))):
        raise ValueError("Require frozen, same-clock development evidence")
    m, e = (a["summary"][k] for k in ("metrics", "effects"))
    if len({(z["views"], z["edits"]) for z in m.values()}) != 1:
        raise ValueError("Require the same realized edit count")

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    vals = dict(Paired=effect(e["paired_vs_fixed"]), Guard=effect(e["guarded_vs_fixed"]),
        Scalar=effect(e["scalar_vs_fixed"]), PairScalar=effect(e["paired_vs_scalar"]),
        GuardScalar=effect(e["guarded_vs_scalar"]), CV=effect(e["guarded_vs_cv"]),
        Oracle=effect(e["oracle_local_vs_fixed"]), Views=m["fixed"]["views"], Edits=m["fixed"]["edits"],
        GuardWithoutHigh=effect(s["contrasts"]["guarded_vs_fixed"]["leave_one_out"]["rdb2"]),
        PairWithoutHigh=effect(s["contrasts"]["paired_vs_fixed"]["leave_one_out"]["rdb2"]),
        WeightedCV=f'{s["contrasts"]["guarded_vs_cv"]["mass_weighted"]:+.3f} pp')
    for arm, tag in (("fixed", "Fixed"), ("old", "Old"), ("scalar", "Reg"), ("paired", "Pair"),
                     ("guarded", "Harm"), ("oracle_local", "Local")):
        for f, name in (("undefined", "Undefined"), ("risk_pass", "Risk"), ("dual_positive", "Joint")):
            vals[tag+name] = m[arm][f]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"A small causal slate menu reaches \SlateQueueTwoDelay{} in hindsight, "
               r"but learning recipients at identical realized counts yields \SlateQueueMain{} "
               r"over a fixed rank (supplement).")
        new = (r"At identical realized counts, direct pairwise recipient training yields \RankClockPaired{} "
               r"over fixed rank, but does not reliably beat same-feature regression or CV (supplement).")
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"ranking.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\RankClock{k}}}{{{v}}}" for k, v in vals.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in paths})
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
    packing = previous.previous.previous.previous.packaging
    if a.source_zip and a.part == "main":
        packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        packing.supplement_package.package(out, a.source_zip, doc)
