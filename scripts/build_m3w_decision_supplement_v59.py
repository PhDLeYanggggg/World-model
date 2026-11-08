"""Add optional CV-to-OLS4 capacity without per-recording solver dumps."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_supplement_v58 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v59"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    root = PUBLIC.parent
    exp = root / "optional_reference_bridge_v1"
    s = exp / "readout.json"
    a = json.loads(s.read_text())
    sha = BASE.cal.digest
    bindings = {Path(v["path"]).name: v["sha256"] for v in a["source_bound"]}
    for k, value in bindings.items():
        f = exp / k
        if (f.exists() or k.endswith("registration.json")) and sha(f) != value:
            raise ValueError("Optional bridge evidence binding changed")
    o, h = [a["radius_comparison"][k] for k in ("opendd", "hannover")]
    if (not a["labels_hindsight_only"] or not a["original_2pct"] or not a["no_training"] or a["confirmation_access"]
            or any(v["unresolved"] for v in (o, h)) or o["views"] != 606 or h["views"] != 9):
        raise ValueError("Require complete optional and clipped oracle comparisons")
    values = {f"{domain}{name}": rows[key] for domain, rows in (("Open", o), ("Han", h))
        for name, key in (("Original", "positive_original"), ("Clipped", "positive_clipped"),
                          ("Union", "positive_unrestricted"), ("Views", "views"))}
    values["Absent"] = o["absent_easy"]
    values["HanRiskOnly"] = a["menus"]["hannover"]["frozen_optional"]["verified_supported_incumbents"]
    d = previous.build(out, style, archive)
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_optional_bridge.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\OptionalBridge{k}}}{{{x}}}"
                                              for k, x in values.items())+"\n")
    d["source_bindings"].update({str(p.relative_to(root)): sha(p) for p in (s, frag)})
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
