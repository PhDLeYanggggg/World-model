"""Bind the OpenDD-trained matched path-choice experiment into the supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v52 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v53"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "matched_path_choice_v1"
    files = [exp / f"{k}.json" for k in ("registration", "head_freeze", "analysis")]
    r, h, a = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    key = sha(files[0])
    if (a["registration"] != key or h["registration"] != key or a["heads"] != sha(files[1])
            or h["target_labels_used"] or r["target_tuning"] or a["target_tuning"]
            or not a["original_controls_exact"] or not a["original_2pct"] or a["confirmation_access"]
            or a["recordings"] != 9 or a["exposed_campaigns"] != 1 or len(h["fit"]) != 3):
        raise ValueError("Require matched controls and frozen OpenDD-only path heads")
    half = a["summary"]["half"]
    full, hand, simple = (half["arms"][k] for k in ("full_agreement", "hindcast", "action_only_agreement"))
    values = dict(CV=f'{full["lower_pp"]["cv"]:+.4f}', Hindcast=f'{full["lower_pp"]["matched_hindcast"]:+.4f}',
        Ablation=f'{half["direct"]["full_agreement_vs_action_only_agreement"]:+.4f}',
        Full=full["risk_passes"], Simple=simple["risk_passes"], Hand=hand["risk_passes"])
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_path_choice.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\PathChoice{k}}}{{{v}}}"
                                              for k, v in values.items())+"\n")
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
