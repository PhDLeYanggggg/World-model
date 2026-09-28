# Why Better Probability Fit Did Not Improve Selection

Posthoc arithmetic on the frozen development readout. This does not modify any model, threshold, action, primary comparison or gate. Sources: summary.json and its hash-bound private evaluation details.

## Benefit and Harm

For each view, error = floor error - selected benefit + selected positive harm. Therefore net gain = harm reduction - lost benefit. Each term below uses the same full-floor denominator. This is not a substitute for the registered selected-risk denominator or the paired ADE comparison denominator.

| New versus control | Lost benefit (pp) | Harm reduction (pp) | Net gain/full floor (pp) |
|---|---:|---:|---:|
| supervised_matched_vs_marginal_matched | 0.0269217 [0.0122322, 0.0435282] | 0.00648741 [0.00216255, 0.0131405] | -0.0204343 [-0.0319997, -0.00956063] |
| marginal_matched_vs_raw_matched | -0.00295359 [-0.00639276, 5.53513e-06] | -0.000966597 [-0.00206887, 0.000258981] | 0.00198699 [-0.000227235, 0.00507391] |
| supervised_matched_vs_raw_matched | 0.0239681 [0.00945249, 0.041685] | 0.00552081 [0.0009513, 0.0128932] | -0.0184473 [-0.0300243, -0.00814984] |
| marginal_joint_vs_raw_joint | -0.523206 [-0.837316, -0.222313] | -0.183142 [-0.303034, -0.0903288] | 0.340064 [0.118521, 0.57632] |
| supervised_joint_vs_raw_joint | 0.648048 [0.331541, 0.997286] | 0.142722 [0.0410423, 0.251742] | -0.505326 [-0.785089, -0.261875] |

Intervals are 3,000 nominal paired-locality bootstrap draws across 12 opened development localities. They are not independent confirmation or simultaneous bounds. The three training seeds and repeated role views are not independent samples.

## Actual Fitting Losses

Each entry is the median across 108 paired fits of the same frozen fitting-monitor component at update 0 or 2,000. Different objectives must not be ranked by their total loss; occurrence and conditional components are unsupervised in the marginal arm. These are fitting diagnostics, not held accuracy or proof of convergence.

| Arm | Component | Initial median | Final median | Heads decreased / 108 |
|---|---|---:|---:|---:|
| marginal | marginal | 0.40023348 | 0.00094040731 | 108 |
| marginal | occurrence | 0.69167089 | 0.84003583 | 23 |
| marginal | conditional | 0.23917755 | 3.2245514 | 3 |
| marginal | supervised | 1.3407868 | 4.2627401 | 9 |
| supervised | marginal | 0.40023348 | 0.0011736908 | 108 |
| supervised | occurrence | 0.69167089 | 0.34593663 | 108 |
| supervised | conditional | 0.23917755 | 0.0027295398 | 108 |
| supervised | supervised | 1.3407868 | 0.35164933 | 108 |

The ratio of final supervised occurrence-loss and marginal-loss medians is 294.74. Both have unit coefficient in the registered objective. This is a ratio of fitting loss values, not a gradient ratio or proof that one loss caused the downstream failure. The next fitting-only diagnostic must measure gradients directly.

## Supported Findings and Open Hypotheses

- The occurrence and conditional-cost fits improve, but the composed signed-risk error does not improve in this readout. Proper occurrence scoring alone is not the downstream objective.
- Matching every query count excludes intervention volume as the sole explanation for the primary negative result. The decomposition quantifies benefit lost versus harm avoided; it does not establish why the shared representation changed.
- A positive mean signed-risk bias can suppress useful admissions. The all-risk score and utility remain frozen, so their residual errors can still produce realized selected-risk violations even when predicted constraints are feasible.
- The sigmoid occurrence probability does not change an individual conditional-risk sign. Its possible value is in relative query weighting, not a separate per-agent safety veto.
- Loss-scale competition, representation sharing and conditional-moment product bias remain hypotheses. This experiment jointly added two auxiliaries, so it cannot identify their separate causal effects.
- Next work should measure component gradient scales and conflict on fitting sources, then preregister a single controlled remedy. Do not tune auxiliary weights, thresholds or eligibility on these held-development outcomes.

No deployment change, new independent data, metric/seconds claim or risk certificate. Stage5C and SMC remain disabled.
