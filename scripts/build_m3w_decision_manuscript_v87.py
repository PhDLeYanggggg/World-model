"""Bind the learned continuation and ranked timing ceiling to the manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v86 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v87"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent/"opendd_credit_value_v2"
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "prefix_registration", "prefix_capacity")]
    r, a, p, c = [json.loads(f.read_text()) for f in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or c["registration"] != sha(paths[2])
            or not p["label_aware_diagnostic"] or not p["exact_original_2pct"]
            or any(z["confirmation_access"] for z in (r, a, p, c))
            or not all(z["original_2pct"] for z in (r, a, c))):
        raise ValueError("Require intact development-only timing evidence")
    m, s, e = (a["summary"][k] for k in ("metrics", "schedule", "effects"))
    n = m["immediate"]["views"]
    if any(sum(z.values()) != n or z.get("unresolved", 0) for z in c["summary"].values()):
        raise ValueError("Ranked capacity is not fully adjudicated")
    if any(m[k]["edits"] != m[k+"_matched"]["edits"] or m[k]["edits"]+s[k]["expired"] != m["immediate"]["edits"]
           for k in ("myopic", "continuation")):
        raise ValueError("Matched evaluation lost released credits")

    def effect(k):
        z = e[k]
        lo, hi = z["interval95"]
        return f'{z["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    values = dict(Fixed=c["summary"]["0"]["qualified"], Delay=c["summary"]["2"]["qualified"], Views=n,
        Myopic=effect("myopic_vs_myopic_matched"), Continue=effect("continuation_vs_continuation_matched"),
        Difference=effect("continuation_vs_myopic"), VersusCV=effect("continuation_vs_cv"))
    for arm, tag in (("immediate", "Now"), ("myopic", "One"), ("continuation", "Tail")):
        for k, v in (("Edits", "edits"), ("Joint", "dual_positive"), ("Undefined", "undefined")):
            values[tag+k] = m[arm][v]
        values[tag+"Unused"] = s[arm].get("expired", 0)
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = (r"The causal value rule yields \NeuralCreditValueEffect{} against immediate spending "
               r"of the same realized credits; the support-priority rule yields "
               r"\NeuralCreditSupportEffect{} (supplement).")
        new = (r"Fixing the recipient ranking lowers the two-query bound to \QueueValueDelay{}. "
               r"Learning continuation value yields \QueueValueDifference{} versus myopic "
               r"spending (supplement).")
        bind, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"continuation.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        bind, receipt = "files", "supplement_bindings.json"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\QueueValue{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(f.relative_to(PUBLIC.parent)): sha(f) for f in paths})
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
    packaging = previous.previous.packaging
    if a.source_zip and a.part == "main":
        packaging.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        packaging.supplement_package.package(out, a.source_zip, doc)
