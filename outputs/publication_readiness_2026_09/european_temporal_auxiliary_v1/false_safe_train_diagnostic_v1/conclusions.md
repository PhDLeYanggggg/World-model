# The Selected Harm Is Underestimated Before Transfer

## What Finished

CREATE job 37812850 completed with exit 0:0 in 7m17s. The numerical replay took
93.706s; these times are distinct. The scheduler briefly remained RUNNING after
the final output, so collection waited for successful accounting rather than
declaring a hang or restarting it. Peak batch RSS was 2,363,580 KiB.

All 216 frozen cost heads were replayed on their original 24 TRAIN packets.
Feature/row/movement, checkpoint and preprocessing hashes match; repeated
inference is exact. No optimizer update, changed threshold, new development
readout or independent-role access occurred. The 432 independent scalar
sum-identity assertions supplement vectorized row checks. The raw field named
`scalar_checks` counts summands as well as identities; it is not a count of
independent assertions. Weights and source arrays stayed on CREATE.

## Findings

| Arm | Known TRAIN selected easy-risk violations /72 | Median selected risk | Median predicted/actual selected easy harm |
|---|---:|---:|---:|
| No auxiliary |58|5.3710%|0.024865|
| Row-mean auxiliary |57|4.5675%|0.037706|
| Temporal auxiliary |52|4.3047%|0.039213|

All three arms underestimate selected easy-harm mass in every one of their 72
source/seed views. For temporal, the predicted/actual harm ratio is 0.705898
on all known TRAIN rows, but only 0.039213 on its selected rows, at the median
view. The action selection exposes a much worse conditional error. The groups
are repeated source/seed views, not 72 independent scenes. These are ratios of
summed positive harm, not percentages of wrong trajectories or FDE degradation.

Temporal's signed excess decomposition, averaging source/seed views inside
locality and then localities equally, is:

```
-1.275936 predicted slack
+9.197277 harm underprediction
-0.457331 budget-weighted reference overprediction
=7.464010 percentage points of excess over the fixed 2% budget.
```

Reference is overpredicted in only 5/72 views; its average signed contribution
is protective, not the main source of excess. Merely lowering the estimated
reference denominator therefore does not address the observed dominant error.
The 1,828 unknown selected temporal occurrences remain unassessed for realized
harm; the known-label failures do not depend on declaring those outcomes safe.

## What This Rules Out And Does Not

The violation exists on TRAIN, so pure unseen-scene shift cannot be its sole
explanation. This does not establish a single causal explanation for all model
failures, nor prove that more fitting, another objective, or richer inputs will
generalize. TRAIN risk is lower than the previous exposed development risk, but
still fails. No model is promoted and the completed temporal readout stays failed.

The diagnosis favors examining the easy-harm regression channel and its
optimization/selection behavior before changing reference calibration or
repeating the unsuccessful threshold and support sweeps. The earlier negative
controls are retained in [prior_controls.md](prior_controls.md).

## One Bounded Repair Candidate

The next candidate replaces only the easy-harm quadratic moment term with a
Poisson-style cost deviance. It is a mean-cost scoring loss on nonnegative
continuous targets, not a claim that trajectory costs are integer counts.
All signed gain/all-risk/easy-risk squared terms and their original weights,
the bounded decoder, the 2% budget, features, roles and fixed update budget must
remain unchanged. Compare with the unchanged no-auxiliary quadratic control;
do not sweep coefficients against the completed readout.

A separately implemented loss primitive has synthetic arithmetic/gradient
checks. In a deliberately tiny-probability decoder case, quadratic error has a
vanishing logit gradient while the stable log-deviance retains a finite gradient.
That establishes an implementation property, **not** evidence that saturation
caused the real-data failure. Real gradients and a matched TRAIN pilot must be
checked before committing a full successor training budget. No successor head
has been trained yet. It remains `not_run`, not a successful repair.

Any successor readout must retain same-count utility comparisons, missing-label
bounds, known-risk failures and undefined support. A model that simply falls
back everywhere does not pass. Independent calibration/confirmation remain
closed, and historical exposed scores remain exploratory.

Image-local detector-silver obs8/pred12 at stride12 raw frames. No metric,
seconds, physical-safety, true-3D or foundation claim. Stage5C and SMC are off.
The research goal and submission-readiness requirements remain unmet.
