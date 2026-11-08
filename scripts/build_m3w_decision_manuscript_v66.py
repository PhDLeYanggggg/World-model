"""Bring the feasibility/discrimination distinction into the main paper."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v50 as main
from scripts import build_m3w_decision_supplement_v65 as previous

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v66"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    paths = [home / "direct_count_value_v1" / f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    paths += [home / k / "analysis.json" for k in ("streaming_menu_capacity_v1", "recording_count_v2")]
    reg, analysis, heads, cap, counts = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    if (analysis["registration"] != sha(paths[0]) or heads["registration"] != sha(paths[0])
            or analysis["heads"] != sha(paths[2]) or len(heads["heads"]) != 42
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (reg, analysis, cap, counts))
            or not reg["no_threshold_tuning"] or not analysis["matched_allocation_no_labels"]
            or not analysis["exact_totals_retrospective_not_online"] or not cap["labels_hindsight_only"]):
        raise ValueError("Require bound model and capacity readouts")
    return paths, analysis["summary"], cap["summary"], counts["summary"]["arms"]["recording"]


def figure(path, u, cap, count):
    from matplotlib import rc_context
    from matplotlib.figure import Figure

    with rc_context({"font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        fig = Figure(figsize=(6.7, 1.86), layout="constrained")
        a, b = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.06, 1.]})
        keys = ("full_total", "full_stream", "nested_total", "nested_stream")
        values = [cap[k]["verified_positive_both"] for k in keys]+[count["dual_positive"]]
        labels = ("Full / total", "Full / online", "Nested / total", "Nested / online", "Learned count")
        a.barh(labels, values, color=["#48769a"]*4+["#b66331"], height=.62)
        a.invert_yaxis()
        for i, x in enumerate(values):
            a.text(x+9, i, str(x), va="center")
        a.set_xlim(0, 606)
        a.set_xlabel("Risk pass and beat both baselines (/606)")
        a.set_title("(a) Label-aware capacity and online readout", loc="left", fontsize=8)
        for i, k in enumerate(("peers_vs_motion", "peers_vs_motion_matched", "peers_vs_proxy")):
            v = u["direct"][k]
            x = v["point"]
            lo, hi = v["interval95"]
            b.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", color="#48769a", capsize=2, markersize=3.5)
        b.axvline(0., color="0.45", ls="--", lw=.8)
        b.set_yticks(range(3), ("vs. motion (online)", "vs. motion (matched)", "vs. prefix proxy"))
        b.set_ylim(2.5, -.5)
        b.set_xlim(-.10, .20)
        b.set_xlabel("Peer model direct gain (pp, 95% location interval)")
        b.set_title("(b) Incremental value of peer context", loc="left", fontsize=8)
        fig.savefig(path, metadata={"Creator": "", "Producer": "", "CreationDate": None, "ModDate": None})


def build(out, style, archive, part):
    paths, a, cap, count = evidence()
    u, m, d = a["online"], a["matched"], a["discrimination"]
    gap = a["direct"]["peers_vs_motion_matched"]
    vals = dict(Strict=cap["nested_stream"]["verified_positive_both"], Views=u["peers"]["views"], Count=count["dual_positive"],
        Proxy=u["past_proxy"]["dual_positive"], Motion=u["motion"]["dual_positive"], Peer=u["peers"]["dual_positive"],
        PeerGain=u["peers_gain_only"]["dual_positive"], MotionGain=u["motion_gain_only"]["dual_positive"],
        MatchedEdits=f'{m["peers"]["edits"]:,}', MatchedPeer=m["peers"]["dual_positive"], MatchedMotion=m["motion"]["dual_positive"],
        Gap=f'{gap["point"]:+.3f}', Low=f'{gap["interval95"][0]:+.3f}', High=f'{gap["interval95"][1]:+.3f}',
        MotionAUC=f'{d["motion"]["gain_auc"]["point"]:.3f}', PeerAUC=f'{d["peers"]["gain_auc"]["point"]:.3f}',
        MotionRisk=f'{d["motion"]["all_excess_auc"]["point"]:.3f}', PeerRisk=f'{d["peers"]["all_excess_auc"]["point"]:.3f}')
    defs = out / "direct_count_values.tex"
    if part == "main":
        doc = main.build(out, style, archive, "main")
        tex = out / "manuscript.tex"
        frag = PUBLIC / "main_direct_count.tex"
        abstract = PUBLIC / "abstract.tex"
        content = tex.read_text()
        i = content.index(r"\begin{abstract}")+len(r"\begin{abstract}")
        j = content.index(r"\end{abstract}", i)
        content = content[:i]+"\n"+abstract.read_text()+content[j:]
        content = BASE.replace_once(content, r"\input{structured_values.tex}", r"\input{structured_values.tex}"+"\n"+r"\input{direct_count_values.tex}")
        tex.write_text(BASE.replace_once(content, r"\section{Limitations and Independent Evidence}",
            r"\input{main_direct_count.tex}"+"\n\n"+r"\section{Limitations and Independent Evidence}"))
        local = out / frag.name
        local.write_text(frag.read_text())
        plot = out / "capacity_discrimination.pdf"
        figure(plot, a, cap, count)
        paths += [frag, abstract]
        binding, manifest, generated = "build_bindings", "source_bindings.json", (tex, defs, local, plot)
    else:
        doc = previous.build(out, style, archive)
        tex = out / "supplement.tex"
        defs = out / "supplement_values.tex"
        frag = PUBLIC / "supplement_direct_count.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        paths += [frag]
        binding, manifest, generated = "files", "supplement_bindings.json", (tex, defs)
    text = "\n".join(rf"\newcommand{{\DirectCount{k}}}{{{v}}}" for k, v in vals.items())+"\n"
    defs.write_text(text if part == "main" else defs.read_text()+text)
    sha = BASE.cal.digest
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in paths})
    for p in generated:
        doc[binding][p.name] = sha(p)
    (out / manifest).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("main", "supplement"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = a.output_dir or DEFAULT / ("build" if a.part == "main" else "supplement")
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == "main":
        main.main.previous.previous.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        previous.package.package(out, a.source_zip, doc)
