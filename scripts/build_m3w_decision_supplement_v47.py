"""Bind optional allocation and its OLS4 ceiling to the working supplement."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v46 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v47"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    exp = home / "optional_allocation_v2"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "capacity_registration", "capacity", "ols_limit")]
    reg, a, r, c, o = [json.loads(p.read_text()) for p in files]
    digest = BASE.cal.digest
    if (a["registration"] != digest(files[0]) or r["parent_registration"] != digest(files[0])
            or c["registration"] != digest(files[2]) or o["registration"] != digest(files[0])
            or not reg["original_scoring_risk_unchanged"] or not o["exact_ols_only_objective_without_risk_or_cv"]
            or any(d["confirmation_access"] for d in (reg, a, r, c, o))):
        raise ValueError("Require the bound source-disjoint allocation evidence")
    risk = a["summary"]["risk"]
    j, p = risk["joint_0"], risk["priced_0"]
    if j["switches"] != p["switches"] or len(a["fits"]) != 27:
        raise ValueError("Require matched allocation controls and all fits")
    effect = a["summary"]["effects"]["joint_0_vs_priced_0"]["completion_lower_pp"]
    lo, hi = effect["interval95"]
    values = dict(Edits=j["switches"], Joint=j["supported_views"], Priced=p["supported_views"],
        Effect=f'{effect["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]',
        NoTwo=o["summary"]["circle"]["negative"][0], NoInherited=o["summary"]["circle"]["negative"][-1],
        Forced=c["counts"]["forced"]["verified_both"], Optional=c["counts"]["optional"]["verified_both"],
        Qualified=sum(f["selected"] is not None for f in a["fits"]))
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_optional_allocation.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(
        rf"\newcommand{{\OptionalAllocation{k}}}{{{v}}}" for k, v in values.items())+"\n")
    d["source_bindings"].update({str(p.relative_to(home)): digest(p) for p in (*files, frag)})
    for p in (tex, defs):
        d["files"][p.name] = digest(p)
    (out / "supplement_bindings.json").write_text(json.dumps(d, indent=2, sort_keys=True)+"\n")
    return d


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    cli.add_argument("--source-zip", type=Path)
    cli.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    cli.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = cli.parse_args()
    receipt = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        previous.package.package(a.output_dir, a.source_zip, receipt)
