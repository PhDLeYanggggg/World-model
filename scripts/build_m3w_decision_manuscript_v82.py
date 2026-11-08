"""Add the fixed-slot shape-choice experiment to the supplementary material."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v81 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v82"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    root = PUBLIC.parent / "donor_shape_choice_v1"
    paths = [root / f"{k}.json" for k in ("registration", "analysis", "head_freeze",
                                          "ceiling_registration", "ceiling")]
    r, a, h, cr, c = [json.loads(p.read_text()) for p in paths]
    sha = BASE.cal.digest
    generator = root.parent / "donor_motion_shape_v1" / "analysis.json"
    if (any(v["registration"] != sha(paths[0]) for v in (a, h)) or a["heads"] != sha(paths[2])
            or c["registration"] != sha(paths[3]) or cr["choice"]["sha256"] != sha(paths[1])
            or r["parent"]["sha256"] != sha(generator)
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (r, a, cr, c))):
        raise ValueError("Require unchanged fixed-slot shape results")
    s = a["summary"]
    p, e, q = s["policies"], s["effects"], c["summary"]
    if not all(v["edits"] == p["anchor"]["edits"] and v["undefined"] == q["no_selected_easy_support"]
               for v in p.values()):
        raise ValueError("Require matched intervention counts and support")

    def effect(k):
        v = e[f"{k}_vs_prefix_match"]
        lo, hi = v["interval95"]
        return f'{v["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'

    v = dict(Fits=len(h["heads"]), Edits=p["anchor"]["edits"], Views=p["anchor"]["views"],
             GainEffect=effect("gain_only"), RiskEffect=effect("risk_aware"),
             OracleEffect=effect("utility_oracle"), Capacity=q["qualified"], Undefined=q["no_selected_easy_support"],
             NoEdit=q["zero_edits_without_support"], NoEasy=q["edited_without_support"],
             UtilityMissing=q["no_dual_utility"], RiskBlocks=q["risk_prevents_win"])
    v["PrefixChoices"] = json.loads(generator.read_text())["summary"]["prefix_chose_extra"][0]
    for name, k in dict(Anchor="anchor", Prefix="prefix_match", Gain="gain_only",
                        Risk="risk_aware", Oracle="utility_oracle").items():
        if k in s["chosen"]:
            v[name+"Choices"] = s["chosen"][k]
        v[name+"Pass"] = p[k]["dual_positive"]
        v[name+"Safety"] = p[k]["risk_pass"]
    if q.get("unresolved", 0) or sum(v[k] for k in ("Capacity", "Undefined", "UtilityMissing", "RiskBlocks")) != v["Views"]:
        raise ValueError("Require resolved outcome partition")

    doc = previous.build(out, style, archive, "supplement")
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "shape_choice.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Shape{k}}}{{{value}}}" for k, value in v.items())+"\n")
    paths.extend([frag, generator])
    doc["source_bindings"].update({str(x.relative_to(PUBLIC.parent)): sha(x) for x in paths})
    doc["files"].update({x.name: sha(x) for x in (tex, defs)})
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
        previous.previous.previous.supplement_package.package(a.output_dir, a.source_zip, doc)
