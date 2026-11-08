"""Add the frozen Hannover transfer and geometric capacity audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v57 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v58"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "hannover_frozen_transfer_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "capacity_registration", "capacity")]
    r, a, cr, c = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or cr["parent"]["sha256"] != sha(files[1])
            or c["registration"] != sha(files[2]) or any(x["confirmation_access"] for x in (r, a, cr, c))
            or not a["no_training"] or not a["no_target_tuning"] or a["formal_confirmation"]
            or a["source_groups"] != 1 or not a["original_2pct"] or not cr["uses_labels_only_for_diagnostic"]):
        raise ValueError("Require the frozen non-confirmatory Hannover transfer")
    h, q = (c["summary"][k] for k in ("half", "quarter"))
    t = a["summary"]["half"]
    values = dict(Targets=f'{sum(r["targets"] for r in a["by_clip"].values()):,}',
        Complete=sum(r["complete12"] for r in a["by_clip"].values()),
        Tangent=f'{t["tangent"]["lower_pp"]["matched_hindcast"]:+.4f}',
        Oracle=f'{t["oracle_utility"]["lower_pp"]["cv"]:+.4f}',
        Cap=f'{h["mean_optimistic_upper"]:+.4f}', Original=f'{c["summary"]["original"]["mean_optimistic_upper"]:+.4f}',
        OLS=f'{h["mean_ols4_lower"]:+.4f}', Risk=q["unknown_alone_exceeds_easy_limit"], Views=q["views"])
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_hannover.tex"
    marker = r"\section*{LLM Usage Statement}"
    src = BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker)
    end = r"\end{thebibliography}"
    src = BASE.replace_once(src, end, "\\bibitem{hannoverrelease}\nH. Cheng. Trajectories in a shared space [data set]. LUIS, 2019. DOI: 10.25835/0094187.\n\n"+end)
    tex.write_text(src)
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\HannoverTransfer{k}}}{{{x}}}"
                                              for k, x in values.items())+"\n")
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
