"""Add the signed-risk and nested calibration evidence."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v60 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v61"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "signed_bridge_risk_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "head_freeze", "admission_registration", "admission")]
    r, a, h, cr, c = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or a["heads"] != sha(files[2]) or c["registration"] != sha(files[3])
            or [v["sha256"] for v in cr["parent"]] != [sha(files[1]), sha(files[2])]
            or any(v["confirmation_access"] for v in (r, a, cr, c)) or not r["hold_gain_and_neural_risk_fixed"]
            or not a["original_2pct"] or not a["no_held_tuning"] or len(h["heads"]) != 147):
        raise ValueError("Require the bound nested signed-risk experiment")
    arms = a["summary"]["arms"]
    v = dict(Direct=arms["direct"]["dual_positive_passes"], Cal=arms["donor_calibrated"]["dual_positive_passes"],
        Edits=f'{arms["donor_calibrated"]["edits"]:,}', Control=arms["calibrated_matched_independent"]["dual_positive_passes"],
        FalseAdmits=c["summary"]["donor_calibrated"]["false_accept"], Oracle=c["summary"]["oracle_bridge_risk"]["dual_positive_passes"])
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_signed_bridge.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\SignedBridge{k}}}{{{x}}}"
                                              for k, x in v.items())+"\n")
    d["source_bindings"].update({str(p.relative_to(root)): sha(p) for p in (*files, frag)})
    for p in (tex, defs):
        d["files"][p.name] = sha(p)
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
