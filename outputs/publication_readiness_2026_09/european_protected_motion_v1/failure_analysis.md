# Failure Analysis and Next Discriminating Test

## What the Registered Comparison Resolves

The earlier negative damping-relative results might have been explained by an
unfair comparison: the neural candidate was protected while raw damping was
not. This experiment applies matching candidate-specific learning and protection
to both. Neural loses all 24 full-pointwise and all 24 hard-subset matched
contrasts with negative conditional intervals. The protected-motion control
therefore rules out that particular explanation in the registered setting.

It does not identify a single cause. Forecast error distributions, cost-target
learnability, source generalization and learned risk conservatism all differ
when the candidate changes. These pipelines must not be described as a pure
forecast-only ablation with all downstream scores held equal.

## Failure Taxonomy

| Issue | Observed evidence | What is not established | Next diagnostic |
|---|---|---|---|
| Candidate value under protection | Neural loses every paired all/hard pointwise contrast; damping easy-neural risk retains about 1.81--1.94% CV gain | Neural architectures can never beat motion baselines | Decompose candidate raw gain/harm and attainable risk-constrained utility on source folds, diagnostic only |
| Utility removed by learned risk | Neural easy-risk intervention is 0.29--1.25%, versus damping 18.77--21.97% without source guard | Every rejected neural switch was useful | Measure held-source utility ranking and gain/harm error jointly; do not tune thresholds on these outcomes |
| Conditional moments imperfect | Example: damping seed17 easy-neural predicts mean harm 0.0865 on locality020 versus observed 0.2851; other localities overpredict harm | A fixed predicted 2% ratio is calibrated actual safety | Separate numerator, event-mass denominator and support errors; use a new frozen design before fitting a correction |
| Residual local safety failure | Unguarded easy-ridge damping fails worst easy in all seeds despite favorable mean easy scores | Mean easy improvement certifies every locality | Keep worst-locality and zero-event checks, not only aggregate ratios |
| Nested producer shift | Neural cost labels use four-site forecasters; held candidate uses the eight-site producer. Damping has no fitted teacher | This shift fully explains the gap | Compare source-excluded, producer-size-matched forecast controls without opening reserved roles |
| Joint contribution unproved | No defined exact-count comparison has a strictly positive conditional interval; many are unsupported | Interactions do not matter in every dataset | Diagnose supported, nonadditive conflict cases before expanding solver trials |
| Rare/missing-label support | Four zero-CV cases in one locality, only 2/12 labels; no zero cases in joint pilot | Successful abstention proves event-specific learning | Retain missingness and absent-support statements; do not delete these targets |
| Numerical instability | Exact infeasibility pruning eliminates current solver failures; old neural bits unchanged | Numerical repair explains predictive gain | Treat as an engineering fix, not method lift |

The new Torch minibatch losses are recorded without smoothing or cherry-picking.
Some final minibatch losses exceed initial losses. Different task scales and
sampling variation prevent a convergence claim from those two points. This
study holds the training budget equal; it does not prove either candidate's
utility/risk head is optimally trained.

## What Worked

The complete nested producer chain, source-only normalization and missing-label
handling reproduce. Fresh damping rollouts match the earlier causal baseline
errors exactly. Utility/risk learning can retain some simple-motion gains while
observing the full-cohort easy and zero checks. The source support guard changes
neither the target population nor labels. The exact-count and same-budget
comparisons are now separated explicitly.

This narrows the defensible method route toward reliable intervention control;
it does not turn a simple predictor into a neural world-model contribution.

## Next Work, Without Reusing Reserved Outcomes

1. Perform a source-only decomposition of lost utility: candidate opportunity,
   gain/harm estimation error, event-mass error, and support abstention. Bind the
   diagnostic to these frozen artifacts and do not select a deployment threshold.
2. Use that decomposition to register one intervention: producer-size matching
   if teacher shift dominates, or a better event-moment target if estimation
   dominates. Train all fixed source-fold controls before reading comparisons.
3. Advance to reserved model selection/calibration only after the method offers
   a defensible strong-control and protection tradeoff. Confirmation remains
   closed until the complete prediction/calibration rule is frozen.

No model-size escalation, extra seed search or easy-risk relaxation is justified
by this readout alone. Generalization, independent safety, physical validity and
paper readiness remain unresolved. No deployment, Stage5C or SMC execution.
