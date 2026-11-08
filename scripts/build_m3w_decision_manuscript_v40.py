"""Add the norm-matched shared-correction diagnostic."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v39 as previous

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v40"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    result_path, mix_path = [home/"shared_correction_audit_v1"/f for f in ("analysis.json", "risk_mixture.json")]
    result, mix = [json.loads(p.read_text()) for p in (result_path, mix_path)]
    if (mix["analysis"]["sha256"] != BASE.cal.digest(result_path)
            or result["confirmation_opened"] or result["risk_rule_changed"] or result["deployment"]
            or result["max_matched_velocity_norm_diff"] > 1e-12):
        raise ValueError("Need the verified same-norm diagnostic only")
    source, target = [result[d]["summary"] for d in ("fitting", "opened_calibration")]
    if (source["views"], target["views"]) != (24, 40):
        raise ValueError("Wrong fitted or opened roster")
    rows = []
    for key, title in (("frozen_hindcast", "Frozen hindcast"), ("magnitude_only", "Magnitude only"),
                       ("shared_removed", "Shared removed")):
        n, m = [d["risk"][key]["supported_views"] for d in (source, target)]
        v = target["effects"][key+"_vs_cv"]["completion_lower_pp"]["point"]
        rows.append(f"{title} & {n} & {m} & {v:+.2f} "+r"\\")
    effect = target["effects"]["shared_removed_vs_magnitude_only"]["completion_lower_pp"]
    g = mix["result"]["frozen_hindcast"]["all"]
    if g["strata"] != ["0", "1"]:
        raise ValueError("Need both and only the measured overlap bins")
    values = dict(Rows="\n".join(rows), Direct=f'{effect["point"]:+.3f}',
        Interval="["+", ".join(f"{x:+.3f}" for x in effect["interval95"])+"]",
        Coverage=f'{100*g["reference_coverage"][0]:.1f}',
        Within=f'{100*g["within"]:+.2f}', Composition=f'{100*g["composition"]:+.2f}')
    return values, {str(p.relative_to(home)): BASE.cal.digest(p) for p in (result_path, mix_path)}


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        old = "on its fitting domain (supplement)."
        new = ("on its fitting domain. Removing shared correction direction at "
               "matched norm did not repair the reversal (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        files = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC/"supplement_shared_correction.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+"\n".join(
            rf"\newcommand{{\SharedCorrection{k}}}{{{v}}}" for k, v in values.items())+"\n")
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
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            previous.previous.previous.previous.prior.main.v22.main.v13.package(out, a.source_zip, doc["build_bindings"])
        else:
            BASE.v22.supp.prior.package(out, a.source_zip, doc)
