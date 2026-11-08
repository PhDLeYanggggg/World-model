"""Update the working paper with the OpenDD CV-tangent result."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v45 as main
from scripts import build_m3w_decision_supplement_v49 as supp

PUBLIC = supp.PUBLIC.parent / "evidence_manuscript_v50"
DEFAULT = supp.DEFAULT.parent / PUBLIC.name
BASE = supp.BASE


def build(out, style, archive, part):
    home = PUBLIC.parent
    exp = home / "opendd_cv_tangent_v1"
    files = [exp / f"{k}.json" for k in ("registration", "analysis", "budget_registration", "budget")]
    r, a, b, c = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (a["registration"] != sha(files[0]) or b["parent"]["sha256"] != sha(files[1])
            or c["registration"] != sha(files[2]) or len(a["models"]) != 42
            or not a["original_2pct_unchanged"] or not b["exact_query_radius_sum"]
            or any(d["confirmation_access"] for d in (r, a, b, c))):
        raise ValueError("Require bound tangent and matched budget development results")
    policy = a["summary"]["policies"]
    budget = c["summary"]
    if len({policy[k]["edits"] for k in ("finite", "tangent", "previous_shape", "half")}) != 1:
        raise ValueError("The held comparison is not count-matched")

    def effect(value, digits=3):
        lo, hi = value["interval95"]
        return f'{value["point"]:+.{digits}f} pp [{lo:+.{digits}f}, {hi:+.{digits}f}]'

    if part == "main":
        doc = main.build(out, style, archive, part)
        tex = out / "manuscript.tex"
        old = "All OpenDD results remain development evidence."
        new = ("At the same recipients and radius, a CV-directional training loss improves "
               "over the prior shape by " + effect(a["summary"]["direct"]["tangent_vs_previous_shape"])
               + f', but passes fewer recording-seed views ({policy["tangent"]["passes"]}/606 versus '
                 f'{policy["previous_shape"]["passes"]}/606). These are development results.')
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        names, manifest, paths = "build_bindings", "source_bindings.json", (tex,)
    else:
        doc = supp.build(out, style, archive)
        tex, defs = out / "supplement.tex", out / "supplement_values.tex"
        frag = PUBLIC / "supplement_cv_tangent.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        values = dict(Edits=policy["tangent"]["edits"], Previous=policy["previous_shape"]["passes"],
            Tangent=policy["tangent"]["passes"], Old=effect(a["summary"]["direct"]["tangent_vs_previous_shape"]),
            Loss=effect(a["summary"]["direct"]["tangent_vs_finite"]),
            Guard=policy["guard"]["passes"], GuardUndefined=policy["guard"]["undefined_easy"],
            Undefined=budget["adaptive_tangent"]["no_easy_in_roster"],
            Adaptive=budget["adaptive_tangent"]["passes"],
            Redistribute=effect(c["paired"]["adaptive_tangent_vs_uniform_tangent"], 4))
        defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Tangent{k}}}{{{v}}}"
            for k, v in values.items())+"\n")
        files.append(frag)
        names, manifest, paths = "files", "supplement_bindings.json", (tex, defs)
    doc["source_bindings"].update({str(p.relative_to(home)): sha(p) for p in files})
    for p in paths:
        doc[names][p.name] = sha(p)
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
    if a.source_zip:
        if a.part == "main":
            main.previous.previous.main_package.package(out, a.source_zip, doc["build_bindings"])
        else:
            supp.package.package(out, a.source_zip, doc)
