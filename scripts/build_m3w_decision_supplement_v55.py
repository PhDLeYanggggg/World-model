"""Include the matched path-replay and exact same-count control."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v54 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v55"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "prefix_path_replay_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "assignment_registration", "assignment")]
    r, a, cr, c = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or cr["parent"]["sha256"] != sha(files[1])
            or c["registration"] != sha(files[2]) or any(x["confirmation_access"] for x in (r, a, cr, c))
            or not a["no_training"] or not a["no_future_inputs"] or not a["original_2pct"]
            or a["target_tuning"] or not cr["exact_expected_utility"]):
        raise ValueError("Require the registered causal replay and matched-count permutation expectation")
    open_dd, tumdot = (a["summary"][k]["half"] for k in ("opendd", "tumdot"))
    x, y = (open_dd["arms"]["linear_prefix"]["lower"][k] for k in ("hindcast", "tangent"))

    def point(v):
        return f'{v["point"]:+.4f}'

    def interval(v):
        lo, hi = v["interval95"]
        return f'{point(v)} [{lo:+.4f}, {hi:+.4f}]'

    v = dict(Hind=point(x), Tangent=point(y), Rank=f'{100*open_dd["rank"]["linear"]["group_mean"]["point"]:.1f}',
        TumLinear=point(tumdot["arms"]["linear_prefix"]["lower"]["cv"]),
        TumAgree=point(tumdot["arms"]["padding_consensus"]["lower"]["cv"]),
        ShuffleLinear=interval(c["summary"]["opendd"]["linear_prefix"]["direct_lower"]),
        ShuffleAgree=interval(c["summary"]["opendd"]["padding_consensus"]["direct_lower"]))
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_prefix_replay.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\PrefixReplay{k}}}{{{x}}}"
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
