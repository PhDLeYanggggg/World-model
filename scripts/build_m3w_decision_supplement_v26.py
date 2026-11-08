"""Add replay-verified observation support and conditional gain to the supplement."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v25 as prior

PUBLIC = prior.PUBLIC.parent / "evidence_manuscript_v26"
DEFAULT = prior.DEFAULT.parent / PUBLIC.name
base = prior.prior


def readouts():
    cal = base.cal
    binding = {}
    result = []
    for name in ("native_observability_v1", "native_conditional_gain_v1"):
        home = PUBLIC.parent / name
        analysis = home / "analysis.json"
        for phase in ("score", "replay"):
            receipt = home / f"{phase}_execution.json"
            cal.verify_receipt(ROOT, cal.read(receipt), analysis)
            binding[f"{name}/{receipt.name}"] = cal.digest(receipt)
        x = cal.read(analysis)
        if x["independent_confirmation"] or not x["original_rule_unchanged"]:
            raise ValueError("Only fitting-source outcomes are allowed in this section")
        binding[f"{name}/{analysis.name}"] = cal.digest(analysis)
        result.append(x)
    obs, gain = result
    if (not gain["frozen_support"] or gain["support_used_in_gain_fit"]
            or obs["switches_per_seed"] != gain["switches_per_seed"]):
        raise ValueError("Expected frozen-support and count-matched evidence")
    c = gain["modes"]["support_matched"]["contrasts"]

    def effect(row):
        return f'{row["point"]:+.3f}' + r"\;[" + ", ".join(f"{v:+.3f}" for v in row["interval95"]) + "]"

    return dict(ObsFits=str(obs["fits"]), CondFits=str(gain["fits"]),
        MotionBrier=f'{obs["brier"]["factor_motion"]["point"]:.3f}',
        QualityBrier=f'{obs["brier"]["factor_quality"]["point"]:.3f}',
        PointError=f'{gain["conditional_mae"]["conditional_point"]["point"]:.3f}',
        RankError=f'{gain["conditional_mae"]["common_step_rank"]["point"]:.3f}',
        MeanError=f'{gain["conditional_mae"]["donor_mean"]["point"]:.3f}',
        QualityPolicy=effect(obs["direct"]["factor_quality_vs_ols4"]["full_grid_lower"]),
        PointSupport=effect(c["conditional_point_vs_support_only"]["full_grid_lower"]),
        RankPoint=effect(c["common_step_rank_vs_conditional_point"]["full_grid_lower"]),
        RankOLS=effect(c["common_step_rank_vs_ols4"]["full_grid_lower"])), binding


def build(out, style, archive):
    macros, bindings = readouts()
    doc = prior.build(out, style, archive)
    frag = PUBLIC / "supplement_support_gain.tex"
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    tex.write_text(base.replace_once(tex.read_text(), r"\section*{LLM Usage Statement}",
        frag.read_text() + "\n\n" + r"\section*{LLM Usage Statement}"))
    defs.write_text(defs.read_text() + "\n".join(
        f"\\newcommand{{\\SupportGain{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    doc["source_bindings"].update(bindings)
    doc["source_bindings"][f"{PUBLIC.name}/{frag.name}"] = base.cal.digest(frag)
    for p in (tex, defs):
        doc["files"][p.name] = base.cal.digest(p)
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=base.v22.STYLE)
    p.add_argument("--archive", type=Path, default=base.v22.supp.base.REPLAY)
    a = p.parse_args()
    result = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        base.v22.supp.prior.package(a.output_dir, a.source_zip, result)
