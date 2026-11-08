"""Add the first-credit result without changing the main paper."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v75 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v76"
DEFAULT = previous.DEFAULT.parent.parent / PUBLIC.name / "supplement"


def build(out, style, archive):
    src = PUBLIC.parent / "first_credit_v2"
    paths = [src / f"{k}.json" for k in ("registration", "analysis")]
    r, a = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (a["registration"] != sha(paths[0]) or not a["no_training"]
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a))):
        raise ValueError("Require the registered first-credit readout")
    s = a["summary"]
    arms, cap = s["arms"], s["one_edit_oracle"]

    def effect(k):
        x = s["advance_minus_anchor_pp"][k]
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    values = dict(Advance=arms["advance"]["dual_positive"], Prefix=arms["advance_prefix"]["dual_positive"],
        Views=arms["anchor"]["views"], Anchor=arms["anchor"]["dual_positive"], Effect=effect("advance"),
        PrefixEffect=effect("advance_prefix"), Short=cap["all_legal"]["views"],
        Rescue=s["short"]["advance"]["dual_positive"], Nominee=cap["nominee"]["views_with_both"],
        Full=cap["all_legal"]["views_with_both"])
    doc = previous.build(out, style, archive)
    tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "credit.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Credit{k}}}{{{v}}}" for k, v in values.items())+"\n")
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
        previous.previous.previous.previous.previous.supplement_package.package(a.output_dir, a.source_zip, doc)
