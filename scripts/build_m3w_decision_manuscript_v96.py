"""Bind nested support supervision and unchanged recording risk transitions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v95 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v96"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    record, audit = timing.legacy.read(timing.RECORD), timing.read_transitions()
    sha = BASE.cal.digest
    rec = PUBLIC.parent/timing.RECORD
    paths = [rec/f"{name}.json" for name in ("registration", "analysis", "head_freeze")]
    r = json.loads(paths[0].read_text())
    for x in r["parents"]:
        f = PUBLIC.parents[2]/x["path"]
        if sha(f) != x["sha256"]:
            raise ValueError("Record-support training lineage changed")
        paths.append(f)
    paths.extend(PUBLIC.parent/timing.MONOTONE/f"{k}.json" for k in ("registration", "analysis"))
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = r"Learning to delay does not reliably help; reranking at the same query also trades support for utility (supplement)."
        new = (r"Adding the support edit makes \RecordSupportNewPass{} undefined views pass "
               r"but moves \RecordSupportLostPass{} passes over the limit: risk is not monotone in edit count.")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"record_support.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    s = record["summary"]
    changes = audit["summary"]["transitions"]
    v = dict(NewPass=changes["undefined_to_pass"], LostPass=changes["pass_to_over_limit"],
             AuditCount=audit["summary"]["active"])
    for key, tag in (("easy", "Easy"), ("local_same_data", "Local"), ("record_no_prefix", "NoPrefix"),
                     ("record_prefix", "Prefix"), ("record_guard", "Guard")):
        v[tag+"Joint"] = s["metrics"][key]["dual_positive"]
        v[tag+"Undefined"] = s["metrics"][key]["undefined"]
        v[tag+"Changed"] = s["changed"].get(key, 0)
    for key, tag in (("record_prefix_vs_local_same_data", "Label"), ("record_prefix_vs_record_no_prefix", "Context"),
                     ("record_guard_vs_easy", "Guard")):
        e = s["effects"][key]
        lo, hi = e["interval95"]
        v[tag+"Effect"] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\RecordSupport{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
