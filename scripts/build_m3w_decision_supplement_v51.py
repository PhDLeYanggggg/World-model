"""Add matched finite-harm continuation and its capacity limit."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v50 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v51"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    exp = home / "opendd_finite_guard_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "capacity_registration", "capacity")]
    r, a, cr, c = [json.loads(p.read_text()) for p in files]
    digest = BASE.cal.digest
    if (a["registration"] != digest(files[0]) or cr["analysis"]["sha256"] != digest(files[1])
            or c["registration"] != digest(files[2]) or len(a["fits"]) != 63
            or not a["original_2pct_unchanged"] or not c["optimum_for_risk_not_utility"]
            or any(d["confirmation_access"] for d in (r, a, cr, c))):
        raise ValueError("Require bound continuation and hindsight capacity")
    primary = a["summary"]["half"]
    policies = primary["policies"]
    original, expanded = (c["summary"]["half"][k] for k in ("original", "expanded"))

    def effect(key):
        x = primary["direct"][key]
        lo, hi = x["interval95"]
        return f'{x["point"]:+.4f} pp [{lo:+.4f}, {hi:+.4f}]'

    v = dict(Edits=f'{a["edits"]:,}', Parent=policies["tangent"]["pass"], Finite=policies["finite_guard"]["pass"],
             Curvature=policies["curvature_guard"]["pass"], FiniteUtility=effect("finite_guard_vs_continued"),
             CurvatureUtility=effect("curvature_guard_vs_finite_guard"), Original=original["pass_views"],
             Expanded=expanded["pass_views"], Absent=expanded["absent"],
             DefinedFailed=expanded["views"]-expanded["absent"]-expanded["pass_views"])
    d = previous.build(out, style, archive, "supplement")
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_finite_guard.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\FiniteGuard{k}}}{{{x}}}"
        for k, x in v.items())+"\n")
    d["source_bindings"].update({str(p.relative_to(home)): digest(p) for p in (*files, frag)})
    for p in (tex, defs):
        d["files"][p.name] = digest(p)
    (out / "supplement_bindings.json").write_text(json.dumps(d, indent=2, sort_keys=True)+"\n")
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    d = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        package.package(a.output_dir, a.source_zip, d)
