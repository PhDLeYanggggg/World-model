"""Render fitting logs only, without loading source arrays or evaluating models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/"outputs/publication_readiness_2026_09/external_cost_bank_v1"


def render(analysis):
    models = analysis["models"]
    keys = sorted({r["view"] for r in models})
    if len(models) != 12 or len(keys) != 6 or not all(r["fit"]["complete"] for r in models):
        raise ValueError("All twelve registered fits must be complete")
    stream = io.StringIO(newline="")
    fields = ["view", "head", "step", "trees", "loss", "gradient_norm", "fitting_fraction_MSE", "nodes"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in models:
        for trace in row["fit"]["trace"]:
            writer.writerow(dict(view=row["view"], head=row["head"], **trace))
    lines = ["# Source-Only Cost-Head Fitting Logs", "",
        "Source: the frozen analysis.json. No external inference or evaluation is performed by this exporter.", "",
        "The neural entries are losses on different sampled minibatches, not a fixed validation set.",
        "Forest entries are draw-count-weighted fitting MSE over sampled source rows. These columns",
        "are not directly comparable performance estimates; no head or seed is selected from them.", "",
        "| Forecast family / seed | Neural step 1 | Neural step 3000 | Forest 16 trees | Forest 128 trees | Fit seconds, both heads |",
        "|---|---:|---:|---:|---:|---:|"]
    for key in keys:
        pair = {r["head"]: r for r in models if r["view"] == key}
        neural, forest = pair["bounded_fraction"]["fit"], pair["matched_fraction_forest"]["fit"]
        lines.append(f"| {key} | {neural['trace'][0]['loss']:.6f} | {neural['trace'][-1]['loss']:.6f} | "
                     f"{forest['trace'][0]['fitting_fraction_MSE']:.6f} | "
                     f"{forest['trace'][-1]['fitting_fraction_MSE']:.6f} | {neural['seconds']+forest['seconds']:.3f} |")
    lines.extend(["", "All six final minibatch losses are below their step-1 values, but that is not",
        "convergence or downstream-lift evidence. The EqMotion seed-43 forest's final fitting MSE",
        "is slightly higher than at sixteen trees. All endpoints are kept regardless of this sign.", "",
        f"Summed fit time: {analysis['summed_fit_seconds']:.6f} seconds. This excludes source loading,",
        "OOF target/feature construction, verification and the already-completed forecasting fits.",
        "The 234 complete log records are in training_loss.csv. No metric/seconds or independent safety claim.", ""])
    return {"training_loss.csv": stream.getvalue(), "training_losses.md": "\n".join(lines)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = REPORT/"analysis.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("replay.json", "independent_verification.json"):
        proof = json.loads((REPORT/name).read_text())
        if proof["analysis_sha256"] != digest or not proof["all_checks_passed"]:
            raise ValueError("Verified analysis required")
    outputs = render(json.loads(path.read_text()))
    for name, value in outputs.items():
        target = REPORT/name
        if args.verify:
            if target.read_bytes() != value.encode():
                raise ValueError("Report mismatch: "+name)
        else:
            target.write_text(value)
    print(json.dumps(dict(files=list(outputs), verify=args.verify, analysis_sha256=digest)))


if __name__ == "__main__":
    main()
