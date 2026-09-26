# What A Fixed-Cap Error Floor Can Establish

H is predicted positive excess trajectory error over the frozen reference,
not an all-agent count. H_E is the easy-event weighted harm moment. The
existing estimator enforces 0 <= H_E <= H <= causal rollout disagreement.
Only the first bound's empirical effect is being diagnosed; all predictions,
trajectories, data roles and event definitions are unchanged.

## Exact Finite-Sample Identity

For a realized nonnegative label y, feasible prediction p and fixed cap c,
the pointwise squared-error minimizer inside [0,c] is q=min(y,c). Expanding
p-y = (p-q)+(q-y) gives

    error = projection floor + distance to projection + boundary cross term
          = (y-q)^2 + (p-q)^2 + 2(y-q)(q-p).

When y <= c the cross term is zero. When y > c, q=c >= p and both factors of
the cross term are nonnegative. The identity remains exact after any fixed
nonnegative weighted average. No new theorem or forecasting algorithm is
claimed: this is an elementary diagnostic of the implemented output range.

The first summary gives equal weight to known positive-envelope fitting rows.
The second uses the prior risk estimator's weights: each known row starts at
1 / (number of fitting localities * known rows in its locality), followed by
the existing positive-envelope restriction and renormalization. This is not
new hard-example oversampling. Empty and unknown rows are not assigned labels.

## Why This Is Not Irreducible Error

Take a future y=0 or 2 with equal probability, constant causal inputs and p=c=1.
p is the exact squared-loss conditional-mean predictor. Its MSE is 1, but its
realized-label projection floor is 0.5. Allowing q to see y artificially
removes future uncertainty. A 50% floor share here diagnoses no conditional
bias at all. This counterexample is checked in the test suite.

The expected nested-cost ordering E[H_E|x] <= E[H|x] is valid. An individual
easy-harm outcome can nevertheless exceed predicted H. It would be incorrect
to remove that expected-cost ordering merely because many observed labels lie
above it. This study does not identify a Bayes noise floor, an attainable
accuracy gain, or a reason to enlarge the deployed model's bounds.

## What The Score-Bin Moments Add

Fitting-only terciles of predicted H/envelope describe where mean(y-H) is
positive or negative. They use no future variable as a binning feature.
They are still descriptive fitting moments: original/cyclic estimators have
in-sample exposure, OOF models use fewer localities, the common event cut uses
all three meta-fitting localities, and views repeat sources and seeds.
No independence, calibration coverage, causal attribution or significance
claim follows from a positive cell. Cell counts are not independent trials.

The useful next decision is whether a cap-only explanation remains adequate,
not whether a label-assisted projection can be promoted. The previously
failed control comparisons remain the model-selection evidence; this study
does not replace them with a simpler pass criterion.
