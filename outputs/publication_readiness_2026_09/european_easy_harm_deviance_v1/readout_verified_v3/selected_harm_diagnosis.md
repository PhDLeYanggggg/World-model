# Selected-Harm Failure Diagnosis

Post-readout descriptive accounting, not a new hypothesis test or independent confirmation.
Inputs are the hash-verified completed six-arm development readout; no model or threshold changed.

| Arm | Harm underpredicted /72 | Undefined harm ratios | Median defined predicted/actual harm | Median defined known risk (%) | Known violations | Completion-upper violations |
|---|---:|---:|---:|---:|---:|---:|
| original | 6 | 32 | 1.904228 | 0.601964 | 4 | 7 |
| additive | 30 | 15 | 0.914493 | 0.844794 | 20 | 42 |
| poisson | 6 | 27 | 2.149372 | 0.509856 | 2 | 11 |
| cost | 3 | 30 | 1.934050 | 0.566794 | 4 | 7 |
| quadratic | 72 | 0 | 0.014273 | 7.382428 | 60 | 72 |
| easy_deviance | 72 | 0 | 0.048558 | 5.517979 | 59 | 71 |

Medians use defined per-view ratios only; undefined counts remain explicit.
Different policies select different populations. A better ratio is not itself a causal effect or policy gain.

| Arm | Predicted slack (pp) | Harm underprediction (pp) | Reference overprediction (pp) | Realized excess (pp) |
|---|---:|---:|---:|---:|
| original | undefined | undefined | undefined | undefined |
| additive | undefined | undefined | undefined | undefined |
| poisson | undefined | undefined | undefined | undefined |
| cost | undefined | undefined | undefined | undefined |
| quadratic | -1.318943 | 13.735302 | -0.499099 | 11.917261 |
| easy_deviance | -1.081269 | 12.295870 | -0.568737 | 10.645864 |

Each term is divided by actual easy-reference mass within the view, then averaged
within locality and across all 12 localities. Any undefined view makes that aggregate undefined.
The signed sum is an accounting identity, not a new theorem or attribution of causal mechanisms.

The deviance repair fails both the matched-quadratic advancement requirement and selected-risk checks.
Known-label risk failures cannot be explained solely by pessimistic missing-outcome bounds.
The complete comparison also shows that lower intervention count does not by itself establish safety.
This diagnosis does not prove that all decision-focused objectives fail, or that features are sufficient.

## Next Controlled Repair

Test a decision-aligned loss on the same frozen forecast candidates, TRAIN roles and sampling budget.
Keep quadratic/deviance controls, the existing 2% positive-easy-harm definition and fixed readout.
Separate optimization of expected intervention utility from independent risk calibration.
First verify gradients and a TRAIN-only pilot, then freeze the full comparison before development readout.
Unknown outcomes must not become zero-cost negatives; independent selection/calibration/confirmation stay closed.
Matched decision-learning training is not_run, not established by this diagnosis.

No deployment change; no metric/seconds, true-3D or foundation-model claim. Stage5C and SMC remain off.
