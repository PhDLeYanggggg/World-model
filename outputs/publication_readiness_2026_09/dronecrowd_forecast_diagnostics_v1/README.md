# Frozen-calibration accuracy and geometry diagnostics

The [secondary protocol](protocol.md) is fixed before the calibration result is available. This analysis does not participate in the threshold decision. It adds available-label ADE and step-12 FDE by difficulty, plus relative-path ADE contrasts at exactly matched per-query intervention counts. All 40 calibration recordings and three seeds must be present; the 30 confirmation groups remain outside the runner.

Nine focused tests passed. On exposed development data, the new pair statistic reproduced the prior additive-versus-joint full-grid lower estimate and interval across 102 recording-seed views to numerical tolerance. Individual masked ADE/FDE was also checked against the prior per-row error routine on six views. These are implementation checks, not external results.

After the registered calibration inference, score and exact replay:

```sh
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_forecast_diagnostics.py score
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_forecast_diagnostics.py replay
```

The executable refuses to access outcomes without verified complete calibration receipts. An aggregate `report.md` is generated after replay. No metric distance, seconds, confirmation, deployment, Stage5C or SMC.
