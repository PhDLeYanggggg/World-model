"""Bind the two-slot support, utility and timing comparison."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v92 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v93"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_two_reserves_v1"
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "ranking_registration",
                                       "ranking_analysis", "timing_registration", "timing_capacity")]
    r, a, rr, q, tr, t = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or q["registration"] != sha(paths[2])
            or t["registration"] != sha(paths[4]) or not t["outcome_aware_only"]
            or any(z["parent"]["sha256"] != sha(paths[1]) for z in (rr, tr))
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a, rr, q, tr, t))):
        raise ValueError("Require bound development-only support evidence")
    s, b = a["summary"], q["summary"]
    if t["summary"]["frozen_query_joint"] != s["metrics"]["oracle_same_query"]["dual_positive"]:
        raise ValueError("Fixed-query witness counts changed")

    def effect(x, digits=3):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.{digits}f} pp [{lo:+.{digits}f}, {hi:+.{digits}f}]'

    e = s["effects"]
    v = dict(Primary=effect(e["two_roles_vs_two_gain_matched"]), Gain=effect(e["gain_vs_cv"]),
        Roles=effect(e["two_roles_vs_cv"]), Match=effect(e["two_gain_matched_vs_cv"]),
        Rerank=effect(b["effects"]["easy"], 5), Bigger=effect(e["two_roles_full_support_vs_two_roles"]),
        AnyTime=t["summary"]["any_query_joint"], ExtraTime=t["summary"]["needs_other_query"])
    for k, tag in (("gain", "Gain"), ("support_only", "Support"), ("two_roles", "Roles"),
                   ("two_gain_matched", "Match"), ("two_roles_full_support", "Bigger"),
                   ("oracle_same_query", "Oracle"), ("constrained_gain", "Rerank")):
        z = b["metrics"][k] if k == "constrained_gain" else s["metrics"][k]
        v[tag+"Edits"], v[tag+"Undefined"], v[tag+"Pass"] = (z[f] for f in ("edits", "undefined", "dual_positive"))
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"Removing incumbent edits gives a one-edit policy with \ShapeSlotFree{} over CV "
               r"but only \ShapeSlotSimplePass{} joint passes (supplement).")
        new = (r"With two matched slots, gain ranking improves \TwoSupportMatch{} over CV; "
               r"easy ranking passes more views (\TwoSupportRolesPass{} vs.\ \TwoSupportMatchPass{}) "
               r"at lower utility (supplement).")
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"two_reserves.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\TwoSupport{k}}}{{{x}}}" for k, x in v.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in paths})
    doc[bind].update({f.name: sha(f) for f in (tex, defs)})
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
