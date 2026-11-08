"""Bind the recording-teacher and extra-edit evidence to the manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v89 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v90"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_record_teacher_v1"
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                       "support_registration", "support_capacity")]
    r, a, h, sr, cap = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or h["registration"] != a["registration"] or a["heads"] != sha(paths[2])
            or cap["registration"] != sha(paths[3]) or sr["parents"][0]["sha256"] != sha(paths[1])
            or sr["changes_main_budget"] or not cap["outcome_aware_only"]
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a, sr, cap))):
        raise ValueError("Require bound development-only teacher and support results")
    m, d, e = (a["summary"][k] for k in ("metrics", "donors", "effects"))
    if len({(z["views"], z["edits"]) for z in m.values()}) != 1:
        raise ValueError("Require identical held edit counts")

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    v = dict(DonorQualified=d["qualified"], Choices=d["example"], Disagrees=d["teacher_disagrees"],
        Direct=effect(e["record_teacher_vs_local_teacher"]), CV=effect(e["record_teacher_vs_cv"]),
        Guard=effect(e["record_teacher_vs_guarded"]))
    for arm, tag in (("fixed", "Fixed"), ("scalar", "Scalar"), ("guarded", "Guarded"),
                     ("local_teacher", "Local"), ("record_teacher", "Record")):
        for f, name in (("undefined", "Undefined"), ("risk_pass", "Risk"), ("dual_positive", "Joint")):
            v[tag+name] = m[arm][f]
    for group, tag in (("no_credits", "Zero"), ("no_reachable_easy", "NoEasy")):
        x = cap["summary"][group]
        v[tag+"Total"], v[tag+"Qualified"], v[tag+"Full"] = x["total"], x["qualified"], x["qualifies_at_1.0"]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"At identical realized counts, direct pairwise recipient training yields \RankClockPaired{} "
               r"over fixed rank, but does not reliably beat same-feature regression or CV (supplement).")
        new = (r"At identical counts, imitating feasible recording assignments yields "
               r"\RecordingTeacherDirect{} versus a local-gain teacher (supplement).")
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"recording_teacher.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\RecordingTeacher{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
    packing = previous.previous.previous.previous.previous.packaging
    if a.source_zip and a.part == "main":
        packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        packing.supplement_package.package(out, a.source_zip, doc)
