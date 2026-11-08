"""Add donor-only calibration and the nonmonotone recording-risk example."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v67 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v68"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    exp = home / "donor_risk_readout_v1"
    paths = [exp / f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    reg, a, h = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (any(x["registration"] != sha(paths[0]) for x in (a, h)) or a["heads"] != sha(paths[2])
            or len(h["triple_ranks"]) != 105 or len(h["pair_moments"]) != 63 or len(h["calibration"]) != 21
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (reg, a))
            or not a["matching_label_blind_not_online"] or not reg["no_threshold_tuning"]):
        raise ValueError("Require bound donor risk and gain results")
    s = a["summary"]
    on, m = s["online"]["arms"], s["matched"]["arms"]
    p = s["online"]["direct"]["calibrated_gain_vs_prefix"]
    easy = s["easy"]
    v = dict(Online=on["calibrated_soft"]["dual_positive"], OnlineFactor=on["factorized"]["dual_positive"],
        Matched=m["calibrated_soft"]["dual_positive"], MatchedFactor=m["factorized"]["dual_positive"], Views=on["factorized"]["views"],
        Fpr=f'{100.*easy["sign"]["calibrated"]["false_positive"]:.2f}', Recall=f'{100.*easy["sign"]["calibrated"]["recall"]:.2f}',
        Brier=f'{easy["brier"]["calibrated"]:.4f}', Prior=f'{easy["brier"]["donor_rate"]:.4f}',
        Gain=f'{p["point"]:+.3f}', GainLow=f'{p["interval95"][0]:+.3f}', GainHigh=f'{p["interval95"][1]:+.3f}')
    doc = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_donor_readout.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\DonorReadout{k}}}{{{v}}}" for k, v in v.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(home)): sha(p) for p in (*paths, frag)})
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
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        previous.previous.previous.package.package(a.output_dir, a.source_zip, doc)
