"""Render all registered arms; never select a deployment winner from readout."""
import json
from pathlib import Path
import platform
import sys
if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 environment required")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest


def num(v): return "undefined" if v is None else f"{v:.6f}"


def main():
    cfg = json.loads((ROOT/"configs/m3w_dimensionless_risk_v1.json").read_text())
    public = ROOT/cfg["reports"]
    a = json.loads((public/"analysis.json").read_text())
    lines = ["# All Registered Results", "", "Design-exposed SDD source-only evidence; no model promotion.",
        "Obs8/pred12 stride12 annotation pixels. Gain is over CV, not seconds/metric prediction.",
        "", "| Action | Policy | ADE gain % | Hard gain % | Worst easy degradation % (site/seed) | Zero-CV harms (row/seed) | Mean selected |", "|---|---|---:|---:|---:|---:|---:|"]
    for action in cfg["actions"]:
        for policy in cfg["policies"]:
            r = a["summary"][action+"__"+policy]
            easy = [-v["gain_percent"] for seed in r["seeds"].values() for v in seed["subsets"]["positive_easy"]["by_scene"].values() if v["gain_percent"] is not None]
            zero = sum(v["zero_CV_harmed"] for v in r["seeds"].values())
            selected = float(np.mean([v["selected"] for v in r["seeds"].values()]))
            lines.append(f"| {action} | {policy} | {num(r['ADE']['equal_scene_gain_percent'])} | {num(r['subsets']['hard']['equal_scene_gain_percent'])} | {num(max(easy))} | {zero} | {selected:.1f} |")
    lines += ["", "Positive-easy degradation excludes exact-zero CV errors; those harms are reported separately.",
              "Repeated seeds/overlapping windows are not independent samples.", "", "## Paired Contrasts", "",
              "Three thousand paired bootstrap resamples of four physical sites; sites are already design-exposed.",
              "All pre-registered contrasts are retained below. No multiple-comparison or independent-confirmation claim.", ""]
    for key, values in a["contrasts"].items():
        lines.extend(["### "+key, "", "```json", json.dumps(values,indent=2), "```", ""])
    lines += ["## Numerical Limitations", "", "```json", json.dumps(a["solver"],indent=2), "```", "",
        "Matched-count contrasts with failed exact-count queries are not fully equal-coverage comparisons.",
        "Unknown-label selections remain unknown; all subset, per-scene/seed and partial-future bounds are in analysis.json.",
        "", "Analysis SHA256: `"+file_digest(public/"analysis.json")+"`."]
    (public/"results.md").write_text("\n".join(lines)+"\n")
    losses = ["# Training Loss and Runtime", "", "All 72 fresh six-output forests use the same fixed budget.",
        "Weighted source fitting MSE is not validation loss or independent predictive evidence.", "",
        "| View | Action | Features | Trees | Fitting mean MSE | Seconds | Unknown draws |", "|---|---|---|---:|---:|---:|---:|"]
    for r in a["fits"]:
        f = r["fit"]
        losses.append(f"| {r['view']} | {r['action']} | {r['arm']} | {f['trees']} | {f['trace'][-1]['fitting_mean_mse']:.7f} | {f['seconds']:.3f} | {f['unknown_sampled']} |")
    losses += ["", "Total fitting-loop seconds: "+str(sum(r["fit"]["seconds"] for r in a["fits"]))+".",
        "This excludes data preparation, provenance hashing, decision solving, evaluation and replay.",
        "Loss traces and held-source six-target MSE are retained in analysis.json."]
    (public/"training_losses.md").write_text("\n".join(losses)+"\n")
    print(json.dumps(dict(report=str(public/"results.md"), fits=len(a["fits"]))))


if __name__ == "__main__": main()
