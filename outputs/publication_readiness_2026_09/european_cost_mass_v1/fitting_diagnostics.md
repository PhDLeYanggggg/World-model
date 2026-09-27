# Fitting Loss Decomposition

Descriptive dependent-view summaries;not held evaluation,selection or independent evidence.
Equal-locality weighting matches fitting. Row weighting is reported separately in JSON.
MSE decomposes into zero-target and positive-target contributions;the per-locality identities are checked.

| Family/arm | Matched mass views | H_easy median mass ratio raw/L2/mass | L2 MSE improves but log-mass worsens | Zero-target LS denominator share |
|---|---:|---|---:|---:|
| full/cost_only | 59/72 | [0.5508030240107118, 0.11757852328321088, 0.9999999999999998] | 60 | 0.9432872570604326 |
| full/cap_aux | 60/72 | [0.601124452011703, 0.19464083407474142, 0.9999999999999997] | 63 | 0.9259969334854848 |
| full/shuffled_aux | 58/72 | [0.5413269809014141, 0.13166948482242738, 0.9999999999999998] | 59 | 0.9459009398684813 |
| motion_only/cost_only | 51/72 | [0.39971830873468633, 0.03109960943359103, 0.9999999999999988] | 64 | 0.9932313442702464 |
| motion_only/cap_aux | 53/72 | [0.3309667347629204, 0.028120470358240918, 0.9999999999999991] | 66 | 0.9920510298506743 |
| motion_only/shuffled_aux | 52/72 | [0.33377753165777835, 0.028697171329056283, 0.9999999999999996] | 65 | 0.9932155420485547 |

No outer outcomes are used in these fits. Slope/cap feasibility failures stay visible.
A restricted origin-L2 estimator plus projection need not preserve mass. This does not refute squared error as a proper mean score.
The fixed projected moment readout tests this specific restriction,not a universal calibration repair.
