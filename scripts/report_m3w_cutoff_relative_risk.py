"""Render every registered source arm and uncertainty without winner selection."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT/"configs/m3w_cutoff_relative_risk_v1.json").read_text())
    public = ROOT/cfg["reports"]; a = json.loads((public/"analysis.json").read_text())
    lines = ["# All Cutoff-Relative Risk Controls", "", "Four already design-exposed SDD sites, three seeds. No deployment selection.",
        "Obs8/pred12 stride12 annotation pixels, not raw t50 or seconds. Gains are over CV, not a strongest-baseline claim.",
        "", "| Action | Policy | ADE gain % | FDE gain % | Hard gain % | Worst positive-easy degradation % | Zero-CV harms, row/seed | Mean selected |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for action in cfg["actions"]:
        for p in a["policies"]:
            r = a["summary"][action+"__"+p]
            easy = max(-v["gain_percent"] for s in r["seeds"].values() for v in s["subsets"]["positive_easy"]["by_scene"].values())
            zero = sum(s["zero_CV_harmed"] for s in r["seeds"].values())
            selected = np.mean([s["selected"] for s in r["seeds"].values()])
            lines.append(f"| {action} | {p} | {r['ADE']['equal_scene_gain_percent']:.6f} | {r['FDE']['equal_scene_gain_percent']:.6f} | {r['subsets']['hard']['equal_scene_gain_percent']:.6f} | {easy:.6f} | {zero} | {selected:.1f} |")
    lines += ["", "The 30 old controls reproduce exactly; they are cached_verified model/decision evidence with fresh reduction checks.",
        "The previous failed matched-count controls remain failures. Three new rules do not establish equal-coverage superiority.",
        "Zero-CV cases are not silently divided by zero. Unknown-label selections and full-grid bounds remain in analysis.json.",
        "", "## All Registered Paired Contrasts", "", "3000 paired four-physical-site resamples, nominal conditional development intervals; not independent confirmation.", ""]
    for key,value in a["contrasts"].items():
        lines += ["### "+key,"","```json",json.dumps(value,indent=2),"```",""]
    lines += ["## Diagnostics", "", "```json",json.dumps(dict(solver=a["solver"],
        unit_predictions_exact=a["metadata_unit_predictions_exact"],unit_probes=a["unit_probes"]),indent=2),"```", ""]
    (public/"results.md").write_text("\n".join(lines))
    lines = ["# Risk-Head Training Loss", "", "36 fresh forests, six bounded targets, same fixed128-tree budget.",
        "Weighted fitting MSE is not independent validation or a safety guarantee.", "",
        "| View | Action | Trees | Fitting MSE | Fitting seconds | Unknown draws |", "|---|---|---:|---:|---:|---:|"]
    for r in a["fits"]:
        f = r["fit"]
        lines.append(f"| {r['view']} | {r['action']} | {f['trees']} | {f['trace'][-1]['fitting_mean_mse']:.7f} | {f['seconds']:.3f} | {f['unknown_sampled']} |")
    lines += ["", "Total fitting-loop seconds: "+str(sum(r["fit"]["seconds"] for r in a["fits"]))+".",
        "This excludes preparation, solving, hashing and replay. All six loss traces and held-source target MSE remain in analysis.json.", ""]
    parent = json.loads((ROOT/"outputs/publication_readiness_2026_09/dimensionless_risk_v1/analysis.json").read_text())
    lines += ["## Matched Held-Source Errors", "",
        "Equal mean across the 12 excluded-site/seed views per action. These are already design-exposed source sites, not independent validation.",
        "Only complete-label rows enter these target errors; the main decision population still includes partial and unknown futures.",
        "Targets, complete-label counts and source views are paired across all three representations. Lower MSE does not certify calibration.", "",
        "| Action | Representation | Overall benefit | Overall harm | Easy harm | Easy benefit | Easy denominator | Easy probability |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for action in cfg["actions"]:
        fresh = [r for r in a["held_source_fit_quality"] if r["action"] == action]
        support = {r["view"]: r["complete_rows"] for r in fresh}
        assert len(support) == 12
        for arm in ("native", "dimensionless", "cutoff_relative"):
            records = fresh if arm == "cutoff_relative" else [
                r for r in parent["held_source_fit_quality"] if r["action"] == action and r["arm"] == arm]
            assert {r["view"]: r["complete_rows"] for r in records} == support
            mse = np.mean([r["held_source_fraction_mse"] for r in records], axis=0)
            lines.append("| "+action+" | "+arm+" | "+" | ".join(f"{v:.7f}" for v in mse)+" |")
    (public/"training_losses.md").write_text("\n".join(lines))
    print(json.dumps(dict(fits=len(a["fits"]),summary_rows=len(a["summary"]))))


if __name__ == "__main__": main()
