# Predicting support did not rescue selective OLS4 replacement

After the OLS4-relative ranker failed at full amplitude, I let it choose 0.25, 0.5 or 1 of the same four causal replacements and abstain if the predicted net gain was nonpositive. I compared direct gain/harm prediction to factorized conditional gain, harm and missingness, with or without past-neighbor information in the support head. The most specific control keeps the fitted conditional heads fixed and substitutes the equal-source fitting prevalence for the learned probability. The earlier [fixed-recipient availability study](../native_visibility_increment_v1/README.md) could not change recipients or amplitude; this one can.

Nine fresh fits use paired initialization and training draws at three seeds. The 69,746 fitting rows from nine sources exclude all fourteen exposed development sources. Forecasts and choices freeze before scoring. Distances are image pixels; eight observations and twelve predictions are annotation steps. Each contrast uses the common observed CV error denominator and 3,000 paired source resamples after seed averaging.

| Contrast | Observed gain (pp) | Completion lower (pp) | 95% paired source CI for lower (pp) |
| --- | ---: | ---: | ---: |
| Neighbor factorization vs unmodified OLS4 | +0.130 | -0.163 | [-0.409, +0.021] |
| Direct, equal count vs unmodified OLS4 | +0.132 | -0.151 | [-0.504, +0.076] |
| Neighbor factorization vs direct, equal count | -0.003 | -0.248 | [-0.579, -0.060] |
| Neighbor factorization vs fixed probability, equal count | +0.006 | -0.267 | [-0.522, -0.101] |
| Neighbor factorization vs ego support, equal count | +0.020 | -0.250 | [-0.513, -0.080] |
| Neighbor factorization vs CV | -3.829 | -17.394 | [-32.909, -6.444] |

Standalone lower bounds against OLS4 are not additive: the direct paired contrasts debit disagreement between the policies being compared on wholly unknown rows. A less negative standalone point against OLS4 does not imply a positive conservative contrast against another policy.

The neighborhood support head has mean AUROC 0.724 and AUPRC 0.0519 against the 0.0225 missing-rate baseline. Brier is 0.02212 versus 0.02228 for fixed probability. It switches 604 wholly unknown rows at the same per-query counts as 950 for fixed probability and 314 for direct. So predicting the released-label mask is possible, but that does not establish better decisions. The neighbor arm activates 9,593/9,675 feasible prior slots (99.15%); 9,582 use the 0.25 amplitude, seven use 0.5, and four use 1. Its summed disagreement debit on unknown selected rows drops from 103,609 to 12,836 image pixels compared with the previous forced full-amplitude policy. Smaller amplitudes explain much of the shrinkage, not a useful abstention rule. No OLS4-floor arm passes the unchanged CV-relative recording risk check; these are diagnostics, not a new risk rule.

This is exposed development, not new external evidence. Scoring and exact replay are `fresh_run`; the training roster and full-amplitude comparison are `cached_verified`. Nine focused tests passed. No external calibration or confirmation readout, threshold tuning, deployment, Stage5C, or SMC.

```sh
runner=scripts/run_m3w_ols_support_selection.py
.venv-pytorch/bin/python "$runner" train
.venv-pytorch/bin/python "$runner" infer
.venv-pytorch/bin/python "$runner" score
.venv-pytorch/bin/python "$runner" replay
```

Resume interrupted training with `train --resume`. [Protocol](protocol.md) | [Results](analysis.json) | [Exact replay](replay_execution.json)
