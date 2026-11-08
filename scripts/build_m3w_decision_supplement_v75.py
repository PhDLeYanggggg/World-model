"""Bind the expiry pricing ablation without changing the main paper."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v74 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v75"
DEFAULT = previous.DEFAULT.parent.parent / PUBLIC.name / "supplement"


def build(out, style, archive):
    src = PUBLIC.parent / "expiry_pricing_v1"
    paths = [src / f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    r, a, h = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (any(v["registration"] != sha(paths[0]) for v in (a, h)) or a["heads"] != sha(paths[2])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a))):
        raise ValueError("Require the frozen expiry comparison")
    s = a["summary"]
    p, d = [s["events"][k] for k in ("priced", "direct")]
    if p["expired"] != d["expired"] or s["arms"]["priced"]["dual_positive"] != s["arms"]["anchor"]["dual_positive"]:
        raise ValueError("Expected the registered comparison, not a new result")
    effect = s["direct"]["priced_vs_direct"]
    lo, hi = effect["interval95"]
    values = dict(Moves=p["moved"], DirectMoves=d["moved"], Count=p["expired"],
        Priced=f'{s["expiry"]["priced"]["point"]:+.6f} pp', Direct=f'{s["expiry"]["direct"]["point"]:+.6f} pp',
        Effect=f'{effect["point"]:+.6f} pp [{lo:+.6f}, {hi:+.6f}]', Joint=s["arms"]["anchor"]["dual_positive"])
    doc = previous.build(out, style, archive)
    tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "expiry.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Expiry{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in (*paths, frag)})
    doc["files"].update({p.name: sha(p) for p in (tex, defs)})
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        previous.previous.previous.previous.supplement_package.package(a.output_dir, a.source_zip, doc)
