"""Add selected-policy calibration and its exact analytic-bank ceiling."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v61 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v62"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "selected_bridge_hurdle_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "head_freeze", "exact_analytic_capacity")]
    r, a, h, c = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or a["heads"] != sha(files[2]) or c["analysis"]["sha256"] != sha(files[1])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a, c))
            or not a["no_held_selection"] or not r["analytic_bank_only"] or not c["labels_hindsight_only"]
            or len(h["heads"]) != 147):
        raise ValueError("Require the bound selected bridge study and hindsight capacity")
    arms = a["summary"]["arms"]
    ci = a["summary"]["direct"]["hurdle_matched_independent"]
    v = dict(Hurdle=arms["support_hurdle"]["dual_positive"], Safety=arms["safety_only"]["dual_positive"],
        Independent=arms["hurdle_matched_independent"]["dual_positive"], Views=arms["support_hurdle"]["views"],
        Point=f'{ci["point"]:+.3f}', Low=f'{ci["interval95"][0]:+.3f}', High=f'{ci["interval95"][1]:+.3f}',
        CV=f'{arms["support_hurdle"]["vs_cv"]["point"]:+.3f}', Capacity=c["counts"]["verified_positive_both"],
        Recordings=c["distinct_recordings"], Absent=c["counts"]["absent_easy"])
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_selected_bridge.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\SelectedBridge{k}}}{{{x}}}"
                                              for k, x in v.items())+"\n")
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
