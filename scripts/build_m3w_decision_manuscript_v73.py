"""Bind the full-ledger comparison and actual-decoder oracle to the draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v72 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v73"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    home = PUBLIC.parent
    src = home / "ledger_wait_v2"
    paths = [src / f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                       "capacity/registration", "capacity/analysis")]
    r, a, h, c, o = [json.loads(p.read_text()) for p in paths]
    digest = BASE.cal.digest
    if (any(x["registration"] != digest(paths[0]) for x in (a, h)) or a["heads"] != digest(paths[2])
            or c["source"]["sha256"] != digest(paths[1]) or o["registration"] != digest(paths[3])
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (r, a, c, o))):
        raise ValueError("Require registered closed-loop supervision and label-aware capacity")
    t, b = a["summary"], o["summary"]
    anchor = t["arms"]["anchor"]["dual_positive"]
    if b["actual_decoder"]["anchor_joint"] != anchor:
        raise ValueError("Actual decoder did not preserve the original roster")
    v = t["direct"]["ledger_risk_vs_local_risk"]
    lo, hi = v["interval95"]
    s = t["label_stats"]
    values = dict(Labels=f'{s["labels"]:,}', Changed=f'{s["actual_target_changed"]:,}',
        Signs=s["gain_sign_changed"], Effect=f'{v["point"]:+.4f} pp [{lo:+.4f}, {hi:+.4f}]',
        Anchor=anchor, Joint=t["arms"]["ledger_risk"]["dual_positive"], Local=t["arms"]["local_risk"]["dual_positive"],
        Restricted=b["exact_token_ceiling"]["verified_positive_both"], Toggle=b["actual_decoder"]["single_toggle_joint_matched"])
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex, defs = out / "manuscript.tex", out / "direct_count_values.tex"
        old = "The causal wait rules do not recover the gain."
        new = (r"Neither greedy nor learned deferral recovers the gain. The actual decoder "
               r"can reach \LedgerToggle{} versus \LedgerAnchor{} with a label-aware, "
               r"count-preserving single change per recording.")
        content = BASE.replace_once(tex.read_text(), old, new)
        bindings, manifest = "build_bindings", "source_bindings.json"
    else:
        tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "ledger.tex"
        marker = r"\section*{LLM Usage Statement}"
        content = BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker)
        paths.append(frag)
        bindings, manifest = "files", "supplement_bindings.json"
    tex.write_text(content)
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Ledger{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(home)): digest(p) for p in paths})
    doc[bindings].update({p.name: digest(p) for p in (tex, defs)})
    (out / manifest).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("main", "supplement"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = a.output_dir or DEFAULT / ("build" if a.part == "main" else "supplement")
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == "main":
        previous.previous.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        previous.previous.supplement_package.package(out, a.source_zip, doc)
