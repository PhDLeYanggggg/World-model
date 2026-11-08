# Frozen-calibration accuracy and geometry diagnostics

The [secondary protocol](protocol.md) was fixed before the calibration result was available. It did not participate in the threshold decision. Full scoring and exact replay completed across all 40 calibration recordings and three seeds; the 30 confirmation groups remain outside the runner.

Available-label CV, OLS4, and ungated joint ADE/FDE are 3.038/5.546, 1.468/2.647, and 2.984/5.446 image pixels. At the same per-query switch counts, joint versus independent has a full-grid lower pair gain of +0.082 pixels (95% paired source-bootstrap interval [+0.067, +0.099]); versus pair-additive it is +0.00084 pixels ([+0.00055, +0.00116]). Only 67 agent windows meet the fixed hard definition, so 24 of 45 source-seed cells cannot estimate hard ADE. The macro hard score is not estimable. This is calibration, not independent validation or a metric claim.

[Full accuracy and pair table](report.md) | [Aggregate results](analysis.json) | [Exact replay](replay_execution.json) | [v24 manuscript](../evidence_manuscript_v24/README.md)

Seventeen related tests passed. Before the new data were scored, the pair statistic reproduced the prior additive-versus-joint full-grid lower estimate and interval on exposed development data across 102 recording-seed views to numerical tolerance. Individual masked ADE/FDE was also checked against the prior per-row error routine on six views. These are separate implementation checks, not external results.

After the registered calibration inference, score and exact replay:

```sh
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_forecast_diagnostics.py score
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_forecast_diagnostics.py replay
```

The executable refuses to access outcomes without verified complete calibration receipts. An aggregate `report.md` is generated after replay. No metric distance, seconds, confirmation, deployment, Stage5C or SMC.
