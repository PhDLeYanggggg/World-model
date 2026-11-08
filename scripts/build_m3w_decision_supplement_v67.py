"""Bind fixed-count component attribution and factorized risk results."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v66 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v67"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    attr = home / "support_value_attribution_v1"
    risk = home / "factorized_count_risk_v1"
    paths = [attr / f"{k}.json" for k in ("registration", "analysis")]+[risk / f"{k}.json" for k in (
        "registration", "analysis", "head_freeze", "risk_readout")]
    ar, a, fr, f, h, r = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or fr["attribution"]["sha256"] != sha(paths[1])
            or any(v["registration"] != sha(paths[2]) for v in (f, h, r)) or f["heads"] != sha(paths[4])
            or r["analysis"]["sha256"] != sha(paths[3]) or len(h["heads"]) != 21
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (ar, a, fr, f, r))
            or not a["matched_not_online"] or not f["matching_label_blind_not_online"]):
        raise ValueError("Require bound support attribution and risk readout")
    s, t = a["summary"], f["summary"]["matched"]
    d = t["factorized_risk"]["direct_vs_signed"]
    q = r["pooled_action_diagnostic"]
    val = dict(Edits=f'{t["signed"]["edits"]:,}', Views=t["signed"]["views"], Base=t["signed"]["dual_positive"],
        Support=s["matched"]["support_oracle"]["dual_positive"], Gain=s["matched"]["gain_oracle"]["dual_positive"],
        Risk=s["matched"]["risk_oracle"]["dual_positive"], All=s["matched"]["all_oracle_local"]["dual_positive"],
        Global=s["ceiling"]["positive_incumbent"], Prefix=s["online"]["past_proxy"]["dual_positive"],
        PrefixRisk=s["online"]["prefix_risk"]["dual_positive"], Factor=t["factorized_risk"]["dual_positive"],
        Point=f'{d["point"]:+.3f}', Low=f'{d["interval95"][0]:+.3f}', High=f'{d["interval95"][1]:+.3f}',
        FalsePositive=f'{100*q["factorized_easy"]["fpr"]:.1f}')
    doc = previous.build(out, style, archive, "supplement")
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_support_factorized.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\FactorRisk{k}}}{{{v}}}" for k, v in val.items())+"\n")
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
        previous.previous.package.package(a.output_dir, a.source_zip, doc)
