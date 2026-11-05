# Complete-Fitting Neural Optimization

I retain the original model, causal features, labels, source roles and
2% risk/support rules, and test complete-fitting optimization. Three
real neural fits and an exact full refit are complete. This is a risk
head diagnostic, not a new trajectory model or a safety certificate.

## Fitting Evidence

| Context | Mean MSE reduction % | Easy MSE reduction % | All/easy sign-cost reduction % | Final gradient max | Stop |
|---|---:|---:|---|---:|---|
| eqmotion_fold1_seed17_critic0 | 31.983 | 32.602 | -8.323/4.209 | 0.001046 | budget_exhausted_not_proven_converged |
| eqmotion_fold0_seed17_critic1 | 37.897 | 42.568 | 1.485/-2.366 | 0.0026122 | budget_exhausted_not_proven_converged |
| eqmotion_fold0_seed17_critic2 | 41.129 | 42.449 | -2.568/-6.341 | 0.027958 | budget_exhausted_not_proven_converged |

Positive reductions mean improvement on fitting data, not held safety.
These losses predict normalized signed all/easy budget-excess risk;
they are not trajectory ADE/FDE. Unknown labels retain original
conservative bounds. Sign costs use observed fitting labels only.

Prospective fitting expansion screen: **False**.
The fixed screen requires >=1% mean-MSE reduction and non-increasing
cost-weighted sign errors in BOTH channels in EVERY fitting context.
It does not change or replace any original evaluation rule.

## Held Evidence

**not_run:** Prospective all3 fitting screen failed; no additional V or H readout.
No new held utility or safety result is inferred from fitting losses.
Earlier held failures remain failures. There is no deployment upgrade.

## Computation And Replay

| Context | Outer calls | Parameter-changing calls | Full gradient passes | LBFGS function evaluations | Seconds |
|---|---:|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | 200 | 200 | 620 | 419 | 54.36 |
| eqmotion_fold0_seed17_critic1 | 200 | 200 | 618 | 417 | 50.35 |
| eqmotion_fold0_seed17_critic2 | 200 | 200 | 613 | 412 | 426.08 |

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 92728 | 68.65 | 6732185600 |
| train | 92869 | 617.61 | 6621839360 |
| evaluate | 93829 | 16.43 | 1000079360 |
| verify | 93881 | 30.50 | 6579666944 |

The evaluate phase may only record not_run; its successful exit does
not imply held evaluation took place. Full-gradient passes include initial
and final-gradient diagnostics and repeated line-search evaluations.
Reaching 200 outer calls is not proof of convergence.

Each context passes actual 3+3 versus 6 resume equivalence. The first full
fit repeats exactly; all initial/final full-fitting metrics replay exactly.
Interrupted line searches cannot overwrite the last completed checkpoint.
This solver/float64/no-decay bundle is not an isolated optimizer comparison.

Original 24-group primary: NA, not passed. Missing support and empty
controls remain NA, without identity certification. No new H/reserved data.
Native obs8/pred12 image pixels; no metric/seconds/true3D/foundation or
human-gold claim. Stage5C and SMC remain disabled.

[Protocol](protocol.md) | [Fitting screen](fitting_screen.json) | [Analysis](analysis.json)
[Error attribution](result_interpretation.md) | [Chinese run guide](operation_zh.md)
