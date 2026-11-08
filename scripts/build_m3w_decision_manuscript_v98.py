"""Bind the matched two-reference direction experiment and site sensitivity."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v79 as packing
from scripts import build_m3w_decision_manuscript_v97 as previous
from scripts import build_m3w_opendd_timing_replay as timing

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v98"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a = timing.legacy.read(timing.DIRECTION)
    sha = BASE.cal.digest
    root = PUBLIC.parent/timing.DIRECTION
    paths = [root/f"{k}.json" for k in ("registration", "analysis", "head_freeze", "sensitivity")]
    b = json.loads(paths[-1].read_text())
    if b["analysis_sha256"] != sha(paths[1]) or b["confirmation_access"] or not b["original_2pct"]:
        raise ValueError("Require verified same-roster sensitivity")
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out/"manuscript.tex", out/"direct_count_values.tex"
        old = r"does not improve the matched comparison (supplement)."
        new = (r"does not improve the matched comparison. At matched radii, learning against "
               r"both baselines beats the CV-only head but not the frozen neural direction (supplement).")
        receipt, bindings = "source_bindings.json", "build_bindings"
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        frag = PUBLIC/"dual_direction.tex"
        paths.append(frag)
        old = r"\section*{LLM Usage Statement}"
        new = frag.read_text()+"\n\n"+old
        receipt, bindings = "supplement_bindings.json", "files"
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    s = a["summary"]
    v = dict(Coview=s["same_query"], SignP=f'{b["two_sided_site_sign_flip_p"]:.4f}')
    for key, tag in (("neural", "Neural"), ("cv_head", "CV"), ("dual", "Dual"),
                     ("guard", "Guard"), ("fixed_oracle", "Oracle")):
        v[tag+"Pass"] = s["metrics"][key]["dual_positive"]
    for tag, e in (("Primary", s["effects"]["dual_vs_cv_head"]),
                   ("Frozen", s["effects"]["dual_vs_neural"]),
                   ("Omit", b["strongest_site_removed"])):
        lo, hi = e["interval95"]
        v[tag] = f'{e["point"]:+.5f} pp [{lo:+.5f}, {hi:+.5f}]'
    defs.write_text(defs.read_text()+"\n"+"\n".join(rf"\newcommand{{\DualDir{k}}}{{{x}}}" for k, x in v.items())+"\n")
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
