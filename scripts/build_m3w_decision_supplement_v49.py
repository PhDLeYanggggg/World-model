"""Bind released-label provenance and the OpenDD held-location control."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v48 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v49"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    moving, opendd = (home / k for k in ("moving_future_support_v1", "opendd_weighted_velocity_v1"))
    files = [p / f"{k}.json" for p in (moving, opendd) for k in ("registration", "analysis")]
    reg, a, oreg, b = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or b["registration"] != sha(files[2])
            or oreg["moving_audit"]["sha256"] != sha(files[1])
            or b["zero_shot"] or len(b["models"]) != 42
            or not a["no_annotation_or_mask_changes"]
            or any(d["confirmation_access"] for d in (reg, a, oreg, b))):
        raise ValueError("Require the bound development-only support and refit evidence")
    m = a["summary"]
    if any(m[k] for k in ("mask_mismatches", "coordinate_mismatches", "past_geometry_row_mismatches")):
        raise ValueError("Raw target reconciliation has unresolved mismatches")
    debt = a["reference_debt_pp_by_wholly_unknown_type"]
    def contrast(label):
        v = b["summary"]["effects"][label]["completion_lower_pp"]
        lo, hi = v["interval95"]
        return f'{v["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'
    values = dict(Steps=f'{m["requested_steps"]:,}', Absent=f'{m["raw_missing_steps"]:,}',
        Keys=f'{m["unique_missing_track_frame_keys"]:,}', NoLater=m["wholly_unknown"]["no_later_same_id"],
        Whole=sum(m["wholly_unknown"].values()), TerminalShare=f'{100*debt["no_later_same_id"]/sum(debt.values()):.2f}',
        OpenRows=f'{b["total_rows"]:,}', OpenFloor=f'{b["reference_debit_floor_pp"]:.3f}',
        OpenCV=contrast("dual_vs_cv"), OpenOLS=contrast("dual_vs_history_ols4_velocity"),
        OpenRisk=b["summary"]["risk"]["dual"]["supported_views"],
        OpenViews=b["summary"]["risk"]["dual"]["required_views"])
    doc = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_raw_support.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\RawSupport{k}}}{{{v}}}"
        for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(home)): sha(p) for p in (*files, frag)})
    for p in (tex, defs):
        doc["files"][p.name] = sha(p)
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


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
