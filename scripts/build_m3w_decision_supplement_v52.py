"""Add a frozen matched-radius transfer experiment and its hindsight bounds."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v51 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v52"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    exp = home / "tumdot_matched_finite_transfer_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "capacity_registration", "capacity")]
    r, a, cr, c = [json.loads(p.read_text()) for p in files]
    digest = BASE.cal.digest
    if (a["registration"] != digest(files[0]) or cr["parent"]["sha256"] != digest(files[1])
            or c["registration"] != digest(files[2]) or not a["historical_replay_exact"]
            or not a["source_is_previously_exposed"] or not cr["labels_only_for_diagnostic"]
            or not r["original_2pct"] or not cr["original_2pct"] or not a["no_target_tuning"]
            or any(d["confirmation_access"] for d in (r, a, cr, c))):
        raise ValueError("Require bound, previously exposed frozen transfer and hindsight audit")
    half, quarter = (a["summary"][k] for k in ("half", "quarter"))
    capacity = c["summary"]["half"]
    opt = capacity["capacity"]["expanded"]
    att = capacity["attribution"]["tangent"]
    if (len(a["recordings"]) != 9 or a["campaigns"] != 1 or half["views"] != 27
            or any(v["passes"] != 27 or v["lower_vs_cv"] >= 0 for v in quarter["policies"].values())):
        raise ValueError("Matched transport scope or quarter-radius findings changed")

    def pp(value):
        return f"{value:+.4f}"

    v = dict(Incumbent=pp(half["direct"]["tangent_vs_incumbent"]),
             Hindcast=pp(half["direct"]["tangent_vs_hindcast"]),
             CV=pp(half["policies"]["tangent"]["lower_vs_cv"]),
             Risk=half["policies"]["tangent"]["passes"],
             Observed=pp(att["observed_pp"]), Partial=f'{att["partial_unknown_pp"]:.4f}',
             Missing=f'{att["wholly_unknown_pp"]:.4f}',
             Witness=opt["risk_optimum"]["joint_positive_cv"],
             WitnessUtility=pp(opt["risk_optimum"]["mean_lower"]["cv"]),
             OptRisk=opt["utility_optimum"]["passed"], OptUtility=pp(opt["utility_optimum"]["mean_lower"]["cv"]))
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_matched_transfer.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\MatchedTransfer{k}}}{{{x}}}"
        for k, x in v.items())+"\n")
    d["source_bindings"].update({str(p.relative_to(home)): digest(p) for p in (*files, frag)})
    for p in (tex, defs):
        d["files"][p.name] = digest(p)
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
