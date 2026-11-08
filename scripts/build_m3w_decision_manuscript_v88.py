"""Add same-clock learned slate comparisons and exact limits to the paper."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v87 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v88"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_slate_queue_v1"
    files = [root/f"{k}.json" for k in ("registration", "analysis", "head_freeze", "clock_registration", "clock_capacity")]
    r, a, h, cr, clock = [json.loads(f.read_text()) for f in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or h["registration"] != a["registration"] or a["heads"] != sha(files[2])
            or clock["registration"] != sha(files[3]) or cr["parent"]["sha256"] != sha(files[1])
            or not cr["preserves_recording_query_counts"]
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (r, a, cr, clock))):
        raise ValueError("Require verified timing evidence, unchanged risk and sealed confirmation")
    s, q = a["summary"], clock["summary"]
    m = s["metrics"]
    n = m["adaptive_queue"]["views"]
    if (m["adaptive_queue"]["edits"] != m["same_clock"]["edits"]
            or m["adaptive_queue"]["edits"] != m["same_credits_immediate"]["edits"]
            or any(sum(x.values()) != n or x.get("unresolved", 0) for x in [q, *s["capacity"].values()])):
        raise ValueError("Require exact, adjudicated controls")

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    val = dict(NoDelay=s["capacity"]["0"]["qualified"], TwoDelay=s["capacity"]["2"]["qualified"],
        ClockCapacity=q["qualified"], Changed=s["same_clock_recipients_changed"],
        Main=effect(s["effects"]["adaptive_queue_vs_same_clock"]),
        Immediate=effect(s["effects"]["adaptive_queue_vs_same_credits_immediate"]),
        Myopic=effect(s["effects"]["adaptive_queue_vs_myopic_queue"]),
        Geometry=effect(s["effects"]["adaptive_queue_vs_no_geometry"]),
        CV=effect(s["effects"]["adaptive_queue_vs_cv"]), Relative=effect(s["pair_relative"]))
    for k, label in (("fixed_now", "FixedNow"), ("adaptive_now", "AdaptiveNow"),
                     ("fixed_queue", "Fixed"), ("myopic_queue", "Current"), ("adaptive_queue", "Joint"),
                     ("no_geometry", "NoGeom"), ("same_clock", "Clock"), ("same_credits_immediate", "Credits")):
        for field, name in (("edits", "Edits"), ("undefined", "Undefined"), ("dual_positive", "Pass")):
            val[label+name] = m[k][field]
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"Fixing the recipient ranking lowers the two-query bound to \QueueValueDelay{}. "
               r"Learning continuation value yields \QueueValueDifference{} versus myopic "
               r"spending (supplement).")
        new = (r"A small causal slate menu reaches \SlateQueueTwoDelay{} in hindsight, "
               r"but learning recipients at identical realized counts yields \SlateQueueMain{} "
               r"over a fixed rank (supplement).")
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"slates.tex"
        files.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\SlateQueue{k}}}{{{v}}}" for k, v in val.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in files})
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
    packing = previous.previous.previous.packaging
    if a.source_zip and a.part == "main":
        packing.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        packing.supplement_package.package(out, a.source_zip, doc)
