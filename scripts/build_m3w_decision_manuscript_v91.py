"""Add the causal reserve experiment and matched support/gain trade-off."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v90 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v91"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_reserve_v1"
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                       "factor_registration", "factor_analysis", "absolute")]
    r, a, h, fr, f, absolute = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or h["registration"] != a["registration"] or a["heads"] != sha(paths[2])
            or f["registration"] != sha(paths[3]) or fr["parent"]["sha256"] != sha(paths[1])
            or absolute["analysis"] != sha(paths[1]) or not absolute["disjoint_edits"]
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a, fr, f, absolute))):
        raise ValueError("Require unchanged-risk reserve evidence")
    s, t = a["summary"], f["summary"]
    for k in ("base", "easy", "gain"):
        if s["metrics"][k] != t["metrics"][k]:
            raise ValueError("Reserve controls did not reproduce")

    def effect(x, digits=3):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.{digits}f} pp [{lo:+.{digits}f}, {hi:+.{digits}f}]'

    v = dict(GainCV=effect(absolute["effects_vs_cv"]["gain"]), EasyBase=effect(s["effects"]["easy_vs_base"]),
        GainBase=effect(s["effects"]["gain_vs_base"]), JointBase=effect(s["effects"]["joint_vs_base"], 4),
        FactorBase=effect(t["effects"]["base"], 4), FactorEasy=effect(t["effects"]["easy_match"], 4),
        FactorGain=effect(t["effects"]["gain_match"], 4), FactorCV=effect(t["effects"]["cv"]),
        JointEasy=effect(s["effects"]["joint_vs_easy_same_query"], 4),
        ZeroEasy=s["cohorts"]["no_credits"]["easy"]["joint"], ZeroFactor=t["cohorts"]["no_credits"]["joint"])
    for arm, tag in (("base", "Base"), ("easy", "Easy"), ("gain", "Gain"), ("joint", "Joint"),
                     ("factor", "Factor"), ("easy_match", "EasyMatch"), ("gain_match", "GainMatch")):
        m = s["metrics"][arm] if arm in s["metrics"] else t["metrics"][arm]
        v[tag+"Extra"] = m["edits"]-s["metrics"]["base"]["edits"]
        v[tag+"Undefined"], v[tag+"Risk"], v[tag+"Pass"] = (m[k] for k in ("undefined", "risk_pass", "dual_positive"))
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"At identical counts, imitating feasible recording assignments yields "
               r"\RecordingTeacherDirect{} versus a local-gain teacher (supplement).")
        new = (r"An easy-triggered reserve raises joint passes from \ReserveSlotBasePass{} to "
               r"\ReserveSlotEasyPass{}, but lowers utility over the incumbent; no arm reliably beats CV (supplement).")
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"reserve.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\ReserveSlot{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
    packing = previous.previous.previous.previous.previous.previous.packaging
    if a.source_zip and a.part == "main":
        packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        packing.supplement_package.package(out, a.source_zip, doc)
