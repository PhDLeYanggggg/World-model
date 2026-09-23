"""Report completed, replay-verified training logs without new model evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/publication_readiness_2026_09/external_predictor_refit_v1"


def summarize(analysis):
    expected = {(family, seed) for family in ("transformer", "eqmotion") for seed in (17, 29, 43)}
    models = analysis["models"]
    if (analysis["scope"] != "fixed_source_predictor_refit_not_external_results"
            or len(models) != 6 or {(m["family"], m["seed"]) for m in models} != expected
            or not analysis["no_model_selection"] or not analysis["source_training_only"]
            or analysis["external_gain_established"] or analysis["deployment"]
            or any(analysis[k] != "not_run" for k in (
                "reserved_source_inference", "independent_calibration", "independent_confirmation"))):
        raise ValueError("Only the complete, fixed, source-training-only matrix may be summarized")
    trials = []
    for model in models:
        fit = model["fit"]
        rows = fit["losses"]
        expected_steps = [1, *range(50, 4001, 50)]
        if (not fit["complete"] or fit["step"] != 4000 or fit["total_draws"] != 256000
                or fit["held_rows_sampled"] != 0 or not math.isfinite(fit["seconds"])
                or fit["seconds"] <= 0 or [r["step"] for r in rows] != expected_steps
                or any(not math.isfinite(r[key]) for r in rows for key in (
                    "loss", "gradient_norm", "learning_rate", "mean_past_normalized_ADE"))):
            raise ValueError("Missing, invalid or mismatched fit/log budget")
        trials.append(dict(family=model["family"], seed=model["seed"],
            fit_seconds=fit["seconds"], optimizer_updates=fit["step"],
            sampled_rows=fit["total_draws"], unique_rows=fit["unique_training_rows"],
            trainable_parameters=model["trainable_parameters"], logged_batches=len(rows),
            first_logged_loss=rows[0]["loss"], final_logged_loss=rows[-1]["loss"],
            first_ten_logged_losses_mean=statistics.mean(r["loss"] for r in rows[:10]),
            last_ten_logged_losses_mean=statistics.mean(r["loss"] for r in rows[-10:]),
            preclip_gradient_above_five_logged_fraction=statistics.mean(r["gradient_norm"] > 5 for r in rows),
            finite_logged_losses=True))
    return dict(scope="sampled_minibatch_training_logs_not_validation_curves", models=len(trials),
        optimizer_updates=sum(t["optimizer_updates"] for t in trials),
        sampled_loss_records=sum(t["logged_batches"] for t in trials),
        summed_fit_seconds=sum(t["fit_seconds"] for t in trials),
        heldout_accuracy_established=False, convergence_established=False, trials=trials)


def main():
    analysis_path = REPORTS / "analysis.json"
    raw = analysis_path.read_bytes()
    analysis = json.loads(raw)
    replay = json.loads((REPORTS / "replay.json").read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if (replay["analysis_sha256"] != digest or replay["models_replayed"] != 6
            or replay["all_checks_passed"] is not True or replay["reserved_source_rows"] != 0):
        raise ValueError("Endpoint replay must precede the training-log summary")
    report = dict(summarize(analysis), analysis_sha256=digest,
        result_source="cached_verified_training_log_aggregation_no_refit")
    (REPORTS / "training_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ["family", "seed", "step", "loss", "gradient_norm", "learning_rate",
              "supported_batch_rows", "mean_past_normalized_ADE"]
    with (REPORTS / "training_loss.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for model in analysis["models"]:
            for row in model["fit"]["losses"]:
                writer.writerow(dict(family=model["family"], seed=model["seed"], **row))
    lines = ["# Source-Only Training Losses", "", "The fixed endpoints were chosen before training, not by these losses.",
        "Loss is a sampled minibatch native-coordinate objective with source-fit normalization.",
        "The first/last summaries average ten logged batches, not epochs or held-out data.",
        "They do not establish convergence, external accuracy, safety or comparative superiority.", "",
        "| Predictor | Seed | Updates | Seconds | First logged loss | Final logged loss | First 10 mean | Last 10 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in report["trials"]:
        lines.append(f'| {t["family"]} | {t["seed"]} | {t["optimizer_updates"]} | {t["fit_seconds"]:.2f} | '
            f'{t["first_logged_loss"]:.6f} | {t["final_logged_loss"]:.6f} | '
            f'{t["first_ten_logged_losses_mean"]:.6f} | {t["last_ten_logged_losses_mean"]:.6f} |')
    lines.extend(["", "Full logged batches: [training_loss.csv](training_loss.csv).",
        "The reported gradient norm is measured before clipping at 5, not the final update norm.",
        "The separate past-normalized ADE debug field is not this objective: very small past",
        "motion denominators can make it large. Neither field is an external performance metric.",
        "All reserved-source predictions, independent calibration and confirmation remain not_run.",
        "", f"Analysis SHA256: `{digest}`.", ""])
    (REPORTS / "training_losses.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
