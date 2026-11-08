"""Bind the fixed-clock support admission experiment to the manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v94 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v95"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/timing.ADMIT
    paths = [root/f"{name}.json" for name in ("registration", "analysis", "head_freeze")]
    r = json.loads(paths[0].read_text())
    a = timing.legacy.read(timing.ADMIT)
    sha = BASE.cal.digest
    if any(sha(PUBLIC.parents[2]/z["path"]) != z["sha256"] for z in r["parents"]):
        raise ValueError("Admission evidence parent changed")
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = r"Learning to delay the support slot does not reliably help (supplement)."
        new = r"Learning to delay does not reliably help; reranking at the same query also trades support for utility (supplement)."
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"support_admission.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    s = a["summary"]
    v = {}
    for tag, arm in (("Easy", "easy"), ("Match", "matched_gain"), ("Gain", "local_gain"),
                     ("Wide", "local_support"), ("Narrow", "restricted_support"), ("Guard", "guarded_support")):
        v[tag+"Joint"] = s["metrics"][arm]["dual_positive"]
        v[tag+"Undefined"] = s["metrics"][arm]["undefined"]
        v[tag+"Changed"] = s["detail"][arm].get("different", 0)
        v[tag+"Outside"] = s["detail"][arm].get("outside_easy", 0)
        key = f"{arm}_vs_easy"
        if key in s["effects"]:
            e = s["effects"][key]
            lo, hi = e["interval95"]
            v[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\Admit{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
