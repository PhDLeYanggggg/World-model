"""Bind the one-edit stopping result and donor utility guard to the manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v80 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v81"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent / "one_edit_stopping_v1"
    paths = [root / f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                         "donor_guard/registration", "donor_guard/analysis")]
    r, a, h, reg, g = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (any(v["registration"] != sha(paths[0]) for v in (a, h)) or a["heads"] != sha(paths[2])
            or g["registration"] != sha(paths[3]) or reg["parents"][0]["sha256"] != sha(paths[1])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a, reg, g))):
        raise ValueError("Require frozen stopping and donor-guard evidence")
    s, m = a["summary"], a["summary"]["metrics"]

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    values = dict(Fits=len(h["heads"]), Learned=m["learned_stop"]["dual_positive"],
        Views=m["immediate"]["views"], Immediate=m["immediate"]["dual_positive"], Fixed=m["fixed_delay"]["dual_positive"],
        Matched=s["common"]["immediate"], FixedMatched=s["common"]["fixed_delay"],
        Full=effect(s["effects"]["immediate"]["full"]), Effect=effect(s["effects"]["immediate"]["matched"]),
        FixedEffect=effect(s["effects"]["fixed_delay"]["matched"]), Fallback=g["totals"]["fallback_sites"],
        Guard=effect(g["lower_vs_immediate_pp"]))
    if g["totals"]["dual_positive"] != values["Immediate"]:
        raise ValueError("Guard claim requires no new joint passes")
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out / "manuscript.tex", out / "direct_count_values.tex"
        old = (r"At one edit per recording, whole-recording event supervision recovers "
               r"\FirstDirect{} joint passes versus \FirstRefit{} from the same head's "
               r"moment readout, below the frozen control's \FirstFrozen{}. At the "
               r"same budget the hindsight ceiling rises from \FirstEarly{} at the "
               r"first eligible query to \FirstAny{} at any eligible query (supplement).")
        new = (r"One-edit event supervision does not beat frozen moments. "
               r"One-edit capacity rises from \FirstEarly{} at the first query to "
               r"\FirstAny{} at any query, yet donor-selected stopping loses "
               r"\StopEffect{} to immediate on \StopMatched{} count-matched "
               r"views (supplement).")
        content = BASE.replace_once(tex.read_text(), old, new)
        bindings, receipt = "build_bindings", "source_bindings.json"
    else:
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        frag = PUBLIC / "stop.tex"
        marker = r"\section*{LLM Usage Statement}"
        content = BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker)
        paths.append(frag)
        bindings, receipt = "files", "supplement_bindings.json"
    tex.write_text(content)
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Stop{k}}}{{{v}}}" for k, v in values.items())+"\n")
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
        previous.previous.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        previous.previous.supplement_package.package(out, a.source_zip, doc)
