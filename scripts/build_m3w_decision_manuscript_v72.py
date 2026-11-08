"""Update the paper with the exact-budget timing and release-value evidence."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v71 as supplement

previous = supplement.previous
BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v72"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    home = PUBLIC.parent
    t = home / "slot_timing_v1"
    w = home / "release_wait_v1"
    paths = [t / f"{k}.json" for k in ("registration", "analysis")] + [w / f"{k}.json" for k in
             ("registration", "analysis", "head_freeze")]
    tr, a, wr, b, h = [json.loads(p.read_text()) for p in paths]
    digest = BASE.cal.digest
    if (a["registration"] != digest(paths[0]) or wr["timing"]["sha256"] != digest(paths[1])
            or any(x["registration"] != digest(paths[2]) for x in (b, h))
            or b["heads"] != digest(paths[4])
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (tr, a, wr, b))):
        raise ValueError("Require hash-bound exact-budget timing and release-value evidence")
    s, q = a["summary"], b["summary"]
    p = s["oracle"]
    fixed, short, long, both = [p[k]["verified_positive_both"] for k in ("fixed", "delay2", "delay8", "move8")]
    quota = s["zero_edit_reachability"]["no_stream_quota"]
    if s["arms"]["anchor"] != q["arms"]["anchor"] or not fixed <= short <= long <= both:
        raise ValueError("Timing evidence no longer shares the frozen reference")
    arm = "predicted_gain_risk"
    gain = q["direct"][arm]
    lo, hi = gain["interval95"]
    values = dict(Fixed=fixed, Short=short, Long=long, Both=both, Quota=quota, Empty=p["fixed"]["zero_edits"],
        Learned=q["arms"][arm]["dual_positive"], Anchor=q["arms"]["anchor"]["dual_positive"],
        Moved=q["actions"][arm]["moved"], Dropped=q["actions"][arm]["dropped"],
        Gain=f'{gain["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]')
    if part == "main":
        doc = previous.build(out, style, archive, "main")
        tex, defs = out / "manuscript.tex", out / "direct_count_values.tex"
        frag = previous.PUBLIC / "main_shape.tex"
        current = frag.read_text()
        old = "roster is unchanged. These are controlled development results."
        new = (r"roster is unchanged. At the same edit counts and radii, permitting two-query "
               r"delay raises fixed-time witnesses from \TimingFixed{} to \TimingShort{}. "
               r"The causal wait rules do not recover the gain. In \TimingQuota{} views, the "
               r"streaming budget cannot fund even one edit.")
        content = BASE.replace_once(tex.read_text(), current, BASE.replace_once(current, old, new))
        bindings, manifest = "build_bindings", "source_bindings.json"
    else:
        doc = supplement.build(out, style, archive)
        tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "timing.tex"
        marker = r"\section*{LLM Usage Statement}"
        content = BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker)
        bindings, manifest = "files", "supplement_bindings.json"
    tex.write_text(content)
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Timing{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(home)): digest(p) for p in (*paths, frag)})
    doc[bindings].update({p.name: digest(p) for p in (tex, defs)})
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
        previous.main_package.package(out, a.source_zip, doc["build_bindings"])
    elif a.source_zip:
        previous.supplement_package.package(out, a.source_zip, doc)
