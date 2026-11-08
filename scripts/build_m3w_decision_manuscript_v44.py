"""Add the recipient objective, predicted capacity, and query-local control."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v43 as previous

PUBLIC = previous.PUBLIC.parent/"evidence_manuscript_v44"
DEFAULT = previous.DEFAULT.parent/PUBLIC.name
BASE = previous.BASE


def evidence():
    home = PUBLIC.parent
    folder = home/"recipient_feasibility_v1"
    files = [folder/"analysis.json", folder/"search_audit.json", folder/"freeze.json",
             home/"recipient_query_fallback_v1"/"analysis.json"]
    a, search, freeze, local = [json.loads(p.read_text()) for p in files]
    if (search["parent"]["sha256"] != BASE.cal.digest(files[2]) or a["fitted_models"] != 54
            or not a["all_arms_same_nominator"] or not freeze["all_choices_frozen_before_score"]
            or any(d["confirmation_access"] for d in (a, search, freeze, local))
            or not local["query_local_only"] or local["summary"]["effects"]["recipient_vs_path"]["known_pp"]["point"] != 0.):
        raise ValueError("Require the frozen source-only recipient and local fallback evidence")
    effect = a["summary"]["effects"]["support_record_vs_prior_record"]["completion_lower_pp"]
    risk = local["summary"]["risk"]["recipient_rank"]
    v = dict(Effect=f'{effect["point"]:+.3f}',
        CI="["+", ".join(f"{x:+.3f}" for x in effect["interval95"])+"]",
        Infeasible=search["counts"]["easy_mass_lower_bound"],
        RiskOnly=a["recipient_bound"]["support_record"]["supported"],
        Moves=risk["switches"], LocalPass=risk["supported_views"],
        LocalGain=f'{local["summary"]["effects"]["recipient_vs_cv"]["completion_lower_pp"]["point"]:+.6f}')
    return v, {str(p.relative_to(home)): BASE.cal.digest(p) for p in files}


def build(out, style, archive, part):
    values, refs = evidence()
    doc = previous.build(out, style, archive, part)
    if part == "main":
        tex = out/"manuscript.tex"
        key, manifest = "build_bindings", "source_bindings.json"
        old = "neural risk heads fail the gate; hindsight path swaps cannot rescue their selected recipients (supplement)."
        new = ("risk heads fail the gate, even with direct recipient-feasibility targets. "
               "Query-local fallback makes few changes and shows no utility lift (supplement).")
        tex.write_text(BASE.replace_once(tex.read_text(), old, new))
        touched = [tex]
    else:
        tex, defs = out/"supplement.tex", out/"supplement_values.tex"
        key, manifest = "files", "supplement_bindings.json"
        frag = PUBLIC/"supplement_recipient_feasibility.tex"
        marker = r"\section*{LLM Usage Statement}"
        tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
        defs.write_text(defs.read_text()+"\n".join(
            rf"\newcommand{{\RecipientFit{k}}}{{{x}}}" for k, x in values.items())+"\n")
        refs[str(frag.relative_to(PUBLIC.parent))] = BASE.cal.digest(frag)
        touched = [tex, defs]
    doc["source_bindings"].update(refs)
    for p in touched:
        doc[key][p.name] = BASE.cal.digest(p)
    (out/manifest).write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("main", "supplement"))
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--source-zip", type=Path)
    cli.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    cli.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = cli.parse_args()
    out = a.output_dir or DEFAULT/("build" if a.part == "main" else "supplement")
    d = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == "main":
            previous.main_package.package(out, a.source_zip, d["build_bindings"])
        else:
            previous.supp_package.package(out, a.source_zip, d)
