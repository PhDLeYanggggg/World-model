"""Add the causal bridge policy and matched-cap failure diagnosis."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v59 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v60"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "reference_bridge_policy_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "head_freeze", "bottleneck_registration", "bottleneck")]
    r, a, h, br, b = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or a["heads"] != sha(files[2]) or b["registration"] != sha(files[3])
            or [v["sha256"] for v in br["parent"]] != [sha(files[1]), sha(files[2])]
            or any(v["confirmation_access"] for v in (r, a, br, b)) or not br["cap_identical_to_learned"]
            or not br["counterfactual_uses_future_for_diagnostics_only"] or not a["original_2pct"] or len(h["heads"]) != 21):
        raise ValueError("Require the bound causal bridge and counterfactuals")
    s = a["summary"]
    j, i = (s["arms"][k] for k in ("joint", "independent"))
    p = s["direct"]["joint_vs_matched_independent"]
    v = dict(Edits=f'{j["edits"]:,}', Independent=f'{i["edits"]:,}', Pass=j["dual_positive_passes"], Views=j["views"],
        Gain=b["counterfactual"]["oracle_gain"]["positive_both"], Risk=b["counterfactual"]["oracle_risk"]["positive_both"],
        Oracle=b["exact"]["verified_positive_both"], Undefined=j["undefined"],
        Direct=f'{p["point"]:+.5f} [{p["interval95"][0]:+.5f}, {p["interval95"][1]:+.5f}]')
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_causal_bridge.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\CausalBridge{k}}}{{{x}}}"
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
