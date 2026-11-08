"""Bind recording-aware count learning and exact-count attribution."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v63 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v64"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "recording_count_v2"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "head_freeze", "matched_diagnostic")]
    r, a, h, m = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (any(v["registration"] != sha(files[0]) for v in (a, h, m)) or a["heads"] != sha(files[2])
            or m["analysis"]["sha256"] != sha(files[1])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a, m))
            or not m["labels_not_used_for_matching"] or m["deployment_inference"]
            or len(h["rank_heads"]) != 63 or len(h["count_heads"]) != 42):
        raise ValueError("Require bound count-head and retrospective matching results")
    arms = a["summary"]["arms"]
    g = a["summary"]["direct"]["recording_vs_local"]
    ma = m["summary"]
    v = dict(LocalN=f'{arms["local"]["edits"]:,}', RecordN=arms["recording"]["edits"],
        Views=arms["recording"]["views"], LocalPass=arms["local"]["dual_positive"],
        RecordPass=arms["recording"]["dual_positive"], Undefined=arms["recording"]["undefined"],
        Gap=f'{g["point"]:+.3f}', GapLow=f'{g["interval95"][0]:+.3f}', GapHigh=f'{g["interval95"][1]:+.3f}',
        MatchedN=ma["scored"]["local"]["edits"], MatchedViews=ma["matched_recordings"],
        Matched=f'{ma["gain"]["point"]:+.3f}', MatchLow=f'{ma["gain"]["interval95"][0]:+.3f}',
        MatchHigh=f'{ma["gain"]["interval95"][1]:+.3f}')
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_recording_count.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\RecordingCount{k}}}{{{x}}}"
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
