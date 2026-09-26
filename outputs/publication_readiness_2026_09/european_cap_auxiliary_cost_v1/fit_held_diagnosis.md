# Descriptive Fitting and Held Error Diagnosis

Secondary diagnostic, not a change to the registered hypothesis or gates.
Each entry summarizes72 dependent seed/locality/assignment views. Counts are not independent trials.

| Inputs / contrast | Fitting median gain (%) | Held median gain (%) | Missing fitting comparisons | Fit positive / held negative | Both negative |
|---|---:|---:|---:|---:|---:|
| full/aux_vs_control | 1.3086695258051821 | 0.4219924779488674 | 0 | 19 | 5 |
| full/aux_vs_shuffled | 2.7885167990469224 | 0.2666140237085611 | 0 | 24 | 4 |
| full/aux_vs_original | None | -0.34639811245887797 | 72 | 0 | 0 |
| motion_only/aux_vs_control | 0.7680787387993739 | -0.04972474022511056 | 0 | 34 | 12 |
| motion_only/aux_vs_shuffled | 0.7330497299718829 | -0.0057299204601348646 | 0 | 34 | 7 |
| motion_only/aux_vs_original | None | 0.012109395258135093 | 72 | 0 | 0 |

Fit/held divergence can be consistent with generalization failure but does not identify its cause.
The original estimator differs in architecture, inputs and objective; only the three new arms are matched.
Legacy original fitting diagnostics have all rows only. Its positive-envelope fitting comparison is not_estimable; denominators are not mixed.
Fixed-batch loss is an optimization diagnostic, not whole-fitting MSE or held performance.
Worst-view details and all supported/missing counts are retained in fit_held_diagnosis.json.
No arm, loss weight, threshold or independent data role is selected by this diagnostic.
No policy/trajectory gain, metric/seconds, true3D or foundation claim. Stage5C/SMC remain off.
