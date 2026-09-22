# Conditional Cost Error Starts During Fitting

Completed 2026-09-22. The diagnostic was committed as `eb3ec702` before its
readout. Two executions finish normally and reproduce identical immutable
analysis SHA256 `9664d1a0628f4b5371636600313c9dea3b167b525c222666010dabd31802bfce`.
This is a fresh diagnostic of cached-verified checkpoints, not new training.
All 36 heads replay 1,581,804 held score rows; 1,080 descriptive records cover
fitting/held populations, training-defined speed and disagreement quantiles,
fixed policy slices and inner fitting sites. Seven scoped tests pass.

## What the Comparison Shows

The previous refit rejected useful high-disagreement deathCircle forecasts.
The new audit asks whether the misranking begins in fitting or only on transfer.
Every quantile boundary uses complete fitting rows with positive past motion or
forecast disagreement. No held outcomes define the bins. Incomplete outcomes
are counted and excluded from cost means, not treated as safe zeros.

In the deathCircle-excluded view, the above-fitting-q99 disagreement stratum has
1,165 complete fitting rows per seed. Its held counterpart has 965, 923 and 909
complete rows, respectively. The table reports seeds 17 / 29 / 43 in order;
costs are annotation pixels, not meters or time-calibrated quantities.

| Quantity | Fitting | Held source |
|---|---|---|
| Realized benefit | 48.97 / 48.70 / 48.94 | 78.21 / 78.56 / 80.53 |
| Fraction-head predicted benefit | 28.67 / 30.49 / 30.66 | 34.71 / 38.59 / 39.58 |
| Realized harm | 10.68 / 11.63 / 11.21 | 22.37 / 23.27 / 23.16 |
| Fraction-head predicted harm | 22.64 / 23.29 / 21.15 | 57.91 / 57.76 / 53.42 |
| Fraction-head net-gain Spearman | .210 / .312 / .353 | .095 / .148 / .146 |
| Native-head net-gain Spearman | .634 / .640 / .643 | .582 / .595 / .634 |

The fraction objective underestimates benefit and overestimates harm already
on fitting rows; transfer amplifies both errors. Its lower overall fraction
MSE than the fitting-only constant control does not establish useful tail
ranking. The native objective ranks this stratum better, but its already fixed
policy fails easy/zero-reference protection elsewhere. Neither result permits
choosing a favorable population or switching to native loss as a success claim.

This local harm overestimation coexists with harm underestimation on the
fraction head's own strict selections. The populations differ. A single global
calibration multiplier is not demonstrated to repair both errors.

## Interpretation and Next Falsifiable Change

The evidence rejects an explanation based solely on held-source feature shift.
It motivates testing loss weighting: fraction MSE weights native squared cost
errors inversely by squared disagreement, while native MSE does not. The next
experiment fixes the intermediate exponent at one, with unchanged architecture,
draws, budget and decision thresholds. It is not a sweep or an already proven
remedy. A registered comparison must still pass accuracy and protection jointly.

In-fit metrics are optimistic. Producer training uses two source sites whereas
the outer candidate uses three; this audit cannot isolate that effect from
covariate or conditional-label shift. Spearman values are descriptive, not
independent statistical confirmation. All four sources are research-exposed;
original closed roles remain unopened. No data role, inference feature, target,
policy, threshold, checkpoint or choice changed during this diagnostic.

The task remains eight observed/twelve predicted sampled annotation steps in
SDD pixel coordinates. Raw-frame t50 is a separate supplement. No metric,
seconds-level, true-3D, foundation, deployment or calibrated-safety claim.
Stage5C and SMC remain unexecuted. This diagnosis alone is not a paper method.
