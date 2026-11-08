"""Add a dual-reference capacity test separating recipients, paths, and count."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v41 as previous

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v42"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    a, b, c, d = [home/"recipient_release_capacity_v1"/f for f in
                   ("analysis.json", "controls.json", "support.json", "amplitude.json")]
    e = home/"recipient_release_crossfit_v2"/"analysis.json"
    cap, control, support, amplitude, fit = [json.loads(p.read_text()) for p in (a, b, c, d, e)]
    if (control["capacity"]["sha256"] != BASE.cal.digest(a) or support["parent"]["sha256"] != BASE.cal.digest(a)
            or cap["independent_confirmation"] or cap["calibration_access"] or control["confirmation_access"]
            or amplitude["parent"]["sha256"] != BASE.cal.digest(a) or fit["independent_confirmation"]
            or amplitude["confirmation_access"] or fit["neural_training"]
            or cap["summary"]["recordings"] != 24 or cap["summary"]["exact_switches"] != 31923):
        raise ValueError("Expected matching fitting-only certificates")
    rows = []
    labels = (("half_exact", "Half, per query"), ("full_exact", "Full added, per query"),
              ("curved_exact", "Curved added, per query"), ("curved_total", "Curved, fixed total"),
              ("half_free", "Half, free count"), ("full_free", "Full added, free count"),
              ("curved_free", "Curved, free count"))
    for k, name in labels:
        row = amplitude["summary"] if k == "full_free" else control["summary"][k]
        rows.append(f'{name} & {row["verified_both"]} & {row["risk_infeasible"]} & '
                    f'{row["negative_upper"]} & {row["unresolved"]} '+r'\\')
    fitted = fit["summary"]
    contrast = fitted["effects"]["predicted_joint_vs_history_ols4_velocity"]["completion_lower_pp"]
    values = dict(Rows="\n".join(rows), HalfExact=control["summary"]["half_exact"]["verified_both"],
        CurvedTotal=control["summary"]["curved_total"]["verified_both"],
        HalfFree=control["summary"]["half_free"]["verified_both"],
        CurvedFree=control["summary"]["curved_free"]["verified_both"],
        NoEasy=support["no_easy_reference_recordings"],
        FitPass=fitted["risk"]["predicted_joint"]["supported_views"],
        FitGain=f'{contrast["point"]:+.3f}',
        FitCI="["+", ".join(f"{x:+.3f}" for x in contrast["interval95"])+"]")
    return values, {str(p.relative_to(home)): BASE.cal.digest(p) for p in (a, b, c, d, e)}


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        old = "recipients (supplement)."
        new = (f"recipients. Releasing recipients at identical query counts gives {values['HalfExact']}/24 "
               f"dual-reference witnesses; freeing counts and expanding paths gives {values['CurvedFree']}/24. "
               f"A causal leave-source-out cost fit recovers {values['FitPass']}/24 passes (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC/"supplement_recipient_release.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+"\n".join(
            rf"\newcommand{{\Release{k}}}{{{v}}}" for k, v in values.items())+"\n")
        refs[str(frag.relative_to(PUBLIC.parent))] = BASE.cal.digest(frag)
        files = [tex, defs]
    doc["source_bindings"].update(refs)
    for f in files:
        doc[key][f.name] = BASE.cal.digest(f)
    (out/manifest).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("main", "supplement"))
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--source-zip", type=Path)
    cli.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    cli.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = cli.parse_args()
    out = a.output_dir or DEFAULT/("build" if a.part == "main" else "supplement")
    d = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            previous.previous.previous.previous.previous.previous.prior.main.v22.main.v13.package(out, a.source_zip, d["build_bindings"])
        else:
            BASE.v22.supp.prior.package(out, a.source_zip, d)
