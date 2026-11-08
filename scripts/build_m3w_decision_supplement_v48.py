"""Attach weighted-velocity and irreducible dual-reference debit evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v47 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v48"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    exp = home / "weighted_velocity_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "debt", "capacity_registration", "capacity")]
    r, a, d, cr, c = [json.loads(p.read_text()) for p in files]
    digest = BASE.cal.digest
    if (a["registration"] != digest(files[0]) or d["analysis"]["sha256"] != digest(files[1])
            or cr["analysis"]["sha256"] != digest(files[1]) or c["registration"] != digest(files[3])
            or not r["original_2pct"] or not d["conservative_protocol_unchanged"]
            or any(v["confirmation_access"] for v in (r, a, d, cr, c))):
        raise ValueError("Require bound weighted-velocity evidence")
    risk = a["summary"]["risk"]
    if risk["dual"]["switches"] != risk["ade"]["switches"] or len(a["fits"]) != 27:
        raise ValueError("Require matched recipients and all fitted cells")

    def contrast(label):
        effect = a["summary"]["effects"][label]["completion_lower_pp"]
        lo, hi = effect["interval95"]
        return f'{effect["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    values = dict(Edits=risk["dual"]["switches"], Dual=contrast("dual_vs_ade"),
        Whole=contrast("whole_dual_vs_history_ols4_velocity"), RiskPass=risk["dual"]["supported_views"],
        Floor=f'{d["sum_debit_floor_pp"]:.3f}', Excess=f'{d["values"]["dual"]["excess_sum_debit"]:.3f}',
        Optimistic=f'{d["values"]["dual"]["triangle_optimistic_joint_sum"]:+.3f}',
        Old=c["original_verified_views"], New=c["new_verified_views"], Unknown=c["summary"]["unresolved"])
    doc = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_weighted_velocity.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(
        rf"\newcommand{{\WeightedVelocity{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(home)): digest(p) for p in (*files, frag)})
    for p in (tex, defs):
        doc["files"][p.name] = digest(p)
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
        previous.previous.package.package(a.output_dir, a.source_zip, d)
