"""Bind the Hannover shared-displacement result into the supplement."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v116 as previous

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v117"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    report = PUBLIC.parent / "hannover_shared_radius_v1"
    analysis, registration = (report / f"{name}.json" for name in ("analysis", "registration"))
    result = json.loads(analysis.read_text())
    if (result["registration"] != hashlib.sha256(registration.read_bytes()).hexdigest()
            or result["confirmation_access"] or not result["hindsight_only"] or not result["original_2pct"]):
        raise ValueError("Require the registered hindsight-only Hannover result")
    rows = result["rows"]
    if len(rows) != 9 or any(row["finite"][b]["status"] != "optimal" for row in rows for b in ("1", "2")):
        raise ValueError("Incomplete Hannover solver outcomes")
    doc = previous.build(out, style, archive, "supplement")
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "hannover_shared_radius.tex"
    anchor = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, frag.read_text() + "\n" + anchor))
    macros = {}
    for j, label in enumerate(("Original", "Double")):
        f = [v["finite"][str(j + 1)] for v in rows]
        u = [v["relaxed"]["all_recipients"]["upper_ols4_pp"][j] for v in rows]
        bound = [v["upper_pp"] for v in f]
        macros[label] = (f"[{min(u):.2f}, {max(u):.2f}] & "
                         f"[{min(bound):.2f}, {max(bound):.2f}] & "
                         f"{sum(v['positive_both'] for v in f)}/{len(f)}")
    needed = [v["relaxed"]["all_recipients"]["minimum_budget_multiple"] for v in rows]
    macros["BreakEven"] = f"{min(needed):.2f}--{max(needed):.2f}"
    defs.write_text(defs.read_text() + "\n" + "\n".join(
        rf"\newcommand{{\HannoverRadius{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): BASE.cal.digest(p)
                                   for p in (analysis, registration, frag)})
    doc["files"].update({p.name: BASE.cal.digest(p) for p in (tex, defs)})
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    args = p.parse_args()
    doc = build(args.output_dir, args.style, args.archive)
    if args.source_zip:
        PACK.supplement_package.package(args.output_dir, args.source_zip, doc)
