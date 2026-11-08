"""Add the equal-budget cross-query ranking ablation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v68 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v69"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    home = PUBLIC.parent
    exp = home / "recording_swap_rank_v1"
    paths = [exp / f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    reg, a, h = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (any(v["registration"] != sha(paths[0]) for v in (a, h)) or a["heads"] != sha(paths[2])
            or len(h["heads"]) != 42 or any(v["confirmation_access"] or not v["original_2pct"] for v in (reg, a))
            or not a["matching_label_blind_not_online"] or not reg["no_threshold_tuning"]):
        raise ValueError("Require bound cross-query swap results")
    s = a["summary"]
    on, matched = s["online"]["arms"], s["matched"]["arms"]
    d = s["matched"]["direct"]["swap_rank_vs_point"]
    v = dict(Edits=f'{matched["point"]["edits"]:,}', Views=matched["point"]["views"],
        OnlinePoint=on["point"]["dual_positive"], OnlineRank=on["swap_rank"]["dual_positive"],
        MatchedPoint=matched["point"]["dual_positive"], MatchedRank=matched["swap_rank"]["dual_positive"],
        MatchedFactor=matched["factorized"]["dual_positive"], Diff=f'{d["point"]:+.3f}',
        Low=f'{d["interval95"][0]:+.3f}', High=f'{d["interval95"][1]:+.3f}',
        PairN=f'{s["swap_pairs"]:,}', Cert=f'{100.*s["certain_swaps"]/s["swap_pairs"]:.2f}',
        AccPoint=f'{100.*s["swap_correct"]["point"]/s["certain_swaps"]:.2f}',
        AccRank=f'{100.*s["swap_correct"]["swap_rank"]/s["certain_swaps"]:.2f}')
    doc = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_recording_swap.tex"
    marker = r"\section*{LLM Usage Statement}"
    text = BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker)
    bib = (r"\bibitem{mandi2022rank} J. Mandi, V. Bucarey, M. Mulamba Ke Tchomba, and T. Guns. "
           r"Decision-focused learning: Through the lens of learning to rank. "
           r"In \emph{ICML}, pages 14935--14947, 2022.")
    tex.write_text(BASE.replace_once(text, r"\end{thebibliography}", bib+"\n\n"+r"\end{thebibliography}"))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\CrossSwap{k}}}{{{x}}}" for k, x in v.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(home)): sha(p) for p in (*paths, frag)})
    for p in (tex, defs):
        doc["files"][p.name] = sha(p)
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        previous.previous.previous.previous.package.package(a.output_dir, a.source_zip, doc)
