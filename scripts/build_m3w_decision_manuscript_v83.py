"""Add the comoving EqMotion producer comparison to the supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v82 as previous
from scripts import build_m3w_decision_manuscript_v79 as packaging

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v83"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    root = PUBLIC.parent / "opendd_comoving_eqmotion_v1"
    files = [root / f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    r, a, h = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (any(v["registration"] != sha(files[0]) for v in (a, h)) or a["heads"] != sha(files[2])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a))):
        raise ValueError("Require the verified exposed EqMotion results")
    s = a["summary"]
    if len(h["heads"]) != len(a["by_site"])*len(r["seeds"])*len(r["arms"]):
        raise ValueError("Incomplete paired training roster")

    def effect(x):
        lo, hi = x["interval95"]
        return f'{x["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    matched, raw = s["matched"], s["raw"]
    if len({(v["edits"], v["views"], v["undefined"]) for v in matched.values()}) != 1:
        raise ValueError("Require unchanged edit and support counts")
    v = dict(Fits=len(h["heads"]), Edits=matched["anchor"]["edits"], Views=matched["anchor"]["views"],
             Undefined=matched["anchor"]["undefined"], PeerEffect=effect(s["raw_effects"]["comoving_peers_vs_comoving_ego"]),
             FrameEffect=effect(s["raw_effects"]["comoving_peers_vs_centered_peers"]),
             EgoVsCV=effect(raw["comoving_ego"]["lower"]["cv"]), EgoVsOLS=effect(raw["comoving_ego"]["lower"]["ols4"]),
             EgoMatched=effect(s["matched_effects"]["comoving_ego"]),
             PrefixPass=matched["prefix_match"]["dual_positive"], AnchorPass=matched["anchor"]["dual_positive"])
    names = dict(CV="cv", OLS="ols4", Centered="centered_peers", Moving="comoving_peers", Ego="comoving_ego")
    for k, arm in names.items():
        v[k+"ADE"] = f'{raw[arm]["complete_ade"]:.3f}'
        v[k+"FDE"] = f'{raw[arm]["endpoint_fde"]:.3f}'
        if arm in matched:
            v[k+"Pass"] = matched[arm]["dual_positive"]
            v[k+"Capacity"] = s["capacity"][arm]["qualified"]
    doc = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "comoving.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\MovingEq{k}}}{{{x}}}" for k, x in v.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): sha(p) for p in (*files, frag)})
    doc["files"].update({p.name: sha(p) for p in (tex, defs)})
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
        packaging.supplement_package.package(a.output_dir, a.source_zip, doc)
