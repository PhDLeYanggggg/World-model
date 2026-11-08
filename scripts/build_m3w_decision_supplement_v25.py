"""Append replay-verified motion and allocation experiments to the v24 supplement."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v24 as prior

PUBLIC = prior.PUBLIC.parent / "evidence_manuscript_v25"
DEFAULT = prior.DEFAULT.parent / PUBLIC.name


def values():
    cal = prior.cal
    bindings = {}

    def checked(name):
        parent = PUBLIC.parent / name
        path = parent / "analysis.json"
        for phase in ("score", "replay"):
            receipt = parent / f"{phase}_execution.json"
            cal.verify_receipt(ROOT, cal.read(receipt), path)
            bindings[f"{name}/{receipt.name}"] = cal.digest(receipt)
        bindings[f"{name}/{path.name}"] = cal.digest(path)
        return cal.read(path), parent

    motion, _ = checked("native_replay_motion_v1")
    alloc, parent = checked("native_expert_risk_allocation_v1")
    path = parent / "regret_diagnostic.json"
    regret = cal.read(path)
    if (motion["independent_confirmation"] or alloc["independent_confirmation"]
            or not alloc["matched_count_each_query"] or not regret["hindsight_only"]
            or regret["score_sha256"] != bindings[f"{parent.name}/analysis.json"]):
        raise ValueError("Require replayed fitting-source results, not confirmation evidence")
    bindings[f"{parent.name}/{path.name}"] = cal.digest(path)

    def effect(x):
        return f'{x["point"]:+.3f}' + r"\;[" + ", ".join(f"{a:+.3f}" for a in x["interval95"]) + "]"

    def contrast(data, left, right):
        return effect(data[f"{left}_vs_{right}"]["full_grid_lower"])

    split = regret["contrasts"]
    return dict(MotionFits=str(motion["fits"]), AllocFits=str(alloc["fit_count"]),
        Switches=f'{alloc["switches_per_seed"]:,}', Queries=f'{alloc["queries"]:,}',
        Motion=contrast(motion["contrasts"], "replay", "ols4"),
        Replay=contrast(motion["contrasts"], "replay", "history"),
        Independent=contrast(alloc["direct"], "independent", "ols4"),
        Minimum=contrast(alloc["direct"], "independent", "minimum"),
        Joint=contrast(alloc["direct"], "joint", "pair_additive"),
        Oracle=effect(alloc["utility"]["oracle"]["full_grid_lower"]),
        Recipient=f'{split["recipient_regret"]["point"]:.3f}',
        Expert=f'{split["action_regret"]["point"]:.3f}',
        CountLow=str(min(alloc["changed_queries_by_seed"]["joint_vs_pair_additive"])),
        CountHigh=str(max(alloc["changed_queries_by_seed"]["joint_vs_pair_additive"]))), bindings


def build(out, style, archive):
    macros, bindings = values()
    doc = prior.build(out, style, archive, "supplement")
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    fragment = PUBLIC / "supplement_expert_allocation.tex"
    tex.write_text(prior.replace_once(tex.read_text(), r"\section*{LLM Usage Statement}",
        fragment.read_text() + "\n\n" + r"\section*{LLM Usage Statement}"))
    defs.write_text(defs.read_text() + "\n".join(
        f"\\newcommand{{\\XAlloc{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    doc["source_bindings"].update(bindings)
    doc["source_bindings"][f"{PUBLIC.name}/{fragment.name}"] = prior.cal.digest(fragment)
    for p in (tex, defs):
        doc["files"][p.name] = prior.cal.digest(p)
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=prior.v22.STYLE)
    p.add_argument("--archive", type=Path, default=prior.v22.supp.base.REPLAY)
    a = p.parse_args()
    result = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        prior.v22.supp.prior.package(a.output_dir, a.source_zip, result)
