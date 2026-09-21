# Forecast Disagreement, Cost Targets and Missing Outcomes

2026-09-21. `fresh_run`: deterministic bounds and training-target statistics.
`cached_verified`: nested forecasts, fixed costs and the legacy stop/strict rule.
No new fit, threshold, policy selection, independent calibration or deployment.

## Complete-Grid Identity

For Euclidean ADE on the same complete twelve-step grid, let B be the baseline
and N the neural forecast. Define D=mean_t ||N_t-B_t|| in annotation pixels.
The triangle inequality gives |ADE(B,Y)-ADE(N,Y)| <= D. For nonnegative realized
benefit and harm, benefit+harm <= D. D is known from past-conditioned forecasts;
the future error and whether a future is exactly CV-predictable are not inputs.

This elementary property verifies on all twelve nested training views. Dividing
complete-grid costs by D yields bounded continuous targets (structural zero when
D=0). Native harm's top1% labels contain34.59-73.70% of squared target energy
across views; the corresponding fraction labels contain4.80-6.90%. Benefit
ranges are41.90-94.42% versus4.15-4.83%. These are training statistics, not proof
of better prediction or a new theoretical contribution. Fraction MSE changes
training weights relative to native MSE and must be tested separately.

47,305 frozen outer query/seed cost predictions exceed full-grid D;3,786 are
inside the existing strict-stop proposal set. Importantly, the older head was
trained on observed-label ADE including partial grids. Thus these counts expose
a mismatch with a complete-grid expected-cost interpretation, not proof that its
original observed-grid target is mathematically impossible. The new experiment
must use a freshly trained complete-only direct control as well as the bounded
head; it must not attribute a supervision-support change to the new architecture.

## Partial Identification, Not Imputation

For observed future steps O and missing steps M, write g_t as baseline error minus
neural error at an observed step, and d_t=||N_t-B_t||. The full-grid gain belongs to

```
[(sum_O g_t - sum_M d_t)/12, (sum_O g_t + sum_M d_t)/12].
```

These pointwise bounds are attainable when missing targets may vary freely; no
smoothness or physical constraint is assumed. The implementation retains actual
observed errors and never invents missing targets. Future masks enter this audit
only after the frozen decisions, not inference or proposal selection.

Three-seed mean full-grid absolute ADE-gain bounds for the fixed strict-stop
control, including all indexed queries:

| Physical Source | Lower | Upper |
| --- | ---: | ---: |
| coupa | -0.007200 | 0.060611 |
| deathCircle | 0.413842 | 0.910773 |
| gates | -0.063434 | 0.540293 |
| hyang | 0.080193 | 0.164557 |

Units are annotation-pixel ADE differences, **not percent gain** and not a
replacement for the primary metric. Unknown baseline future error prevents
identifying the same full-population relative-percentage estimand from these
intervals. These are deterministic completion ranges on explored sources, not
population confidence intervals or independent generalization guarantees.

There are289 selected incomplete query/seed instances still compatible with an
exactly CV-correct complete future and positive replacement harm. Complete-label
zero observed harms therefore cannot certify full-population zero harm. This
does not establish that all289 really are harmed; their future is unobserved.

## Next Fixed Experiment and Remaining Data Gap

Test direct-native cost regression, disagreement-bounded native regression, and
bounded-fraction regression with the same features, architecture, draws, complete
supervision and fixed budget. Separate parameterization from weighting. Retain
strict zero-CV evaluation, positive-easy<=2% diagnostics, matched-count controls,
unknown outcomes and all seeds. Do not evaluate success by cost MSE alone.

Existing calibration records still assign no independent calibration scenes.
The inspected DUT intake is diagnostic: two sites, one duplicate-annotation clip
quarantined, unresolved original-use/exposure/role conditions. It is not silently
promoted to confirmation. No new external source, independent split or CREATE
job is established here. More overlapping rows do not fix that missing evidence.

No change to8observed/12predicted native-coordinate ADE/FDE, no new metric or
seconds claim, no Stage5C/SMC. Tests cover exact-zero handling, missing-outcome
sharpness, structural agreement, mask isolation and target-bound violations.
