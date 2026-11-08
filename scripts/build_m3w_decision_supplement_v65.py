"""Bind the streaming menu audit and budget-aligned teacher experiment."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v64 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v65"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    cap = root / "streaming_menu_capacity_v1"
    train = root / "token_count_teacher_v1"
    files = [cap / f"{k}.json" for k in ("registration", "analysis")]+[train / f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    cr, c, tr, t, h = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (c["registration"] != sha(files[0]) or t["registration"] != sha(files[2])
            or t["heads"] != sha(files[4]) or tr["capacity"]["sha256"] != sha(files[1])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (cr, c, tr, t))
            or not c["labels_hindsight_only"] or not tr["no_held_tuning"] or len(h["heads"]) != 42):
        raise ValueError("Require the registered capacity and matched teacher studies")
    s, z = c["summary"], t["summary"]
    g = t["direct_token_vs_projected"]
    v = dict(FullTotal=s["full_total"]["verified_positive_both"], FullToken=s["full_stream"]["verified_positive_both"],
        NestedTotal=s["nested_total"]["verified_positive_both"], NestedToken=s["nested_stream"]["verified_positive_both"],
        Views=z["token_exact"]["views"], Projected=z["projected"]["dual_positive"], Exact=z["token_exact"]["dual_positive"],
        Gap=f'{g["point"]:+.3f}', Low=f'{g["interval95"][0]:+.3f}', High=f'{g["interval95"][1]:+.3f}')
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_streaming_teacher.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\StreamingTeacher{k}}}{{{x}}}"
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
