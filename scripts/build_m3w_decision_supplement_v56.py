"""Include the static-road experiment with support-matched assignment controls."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v55 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v56"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "opendd_map_direction_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "support_registration", "support")]
    r, a, sr, s = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or sr["parent"]["sha256"] != sha(files[1])
            or s["registration"] != sha(files[2]) or any(x["confirmation_access"] for x in (r, a, sr, s))
            or not a["no_training"] or not r["no_future_inputs"] or not a["original_2pct"]
            or not a["control_replay_exact"] or not sr["excludes_future_labels_from_strata"]):
        raise ValueError("Require the registered matched road-context readout")
    primary = a["summary"]["half"]
    v = primary["arms"]["map_continuation"]

    def effect(d):
        lo, hi = d["interval95"]
        return f'{d["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'

    values = dict(Mapped=f'{primary["support"]["fully_mapped"]:,}', Edited=f'{primary["support"]["edited"]:,}',
                  Tangent=effect(v["lower"]["tangent"]), Pass=v["risk"]["pass_views"],
                  Overall=effect(s["summary"]["recording"]["map_continuation"]["direct_lower"]),
                  State=effect(s["summary"]["map_state"]["map_continuation"]["direct_lower"]))
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_road_context.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\RoadContext{k}}}{{{x}}}"
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
