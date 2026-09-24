# Positive Harm versus Net Easy Risk

## What Is Being Changed

Let `b` be CV ADE and `a` the candidate ADE in native annotation pixels. For a
source-fitted cutoff `C`, define `E = 1[b <= C]`. This event includes zero-CV
cases; they are separately reported because percentage degradation is undefined
there. Let `d` be the mean distance between the two predicted trajectories.
The triangle inequality bounds both `max(a-b,0)` and `max(b-a,0)` by `d` on the
same complete set of waypoints.

The new four-output forest learns bounded labels:

```text
u+ = E * max(a-b, 0) / d
u- = E * max(b-a, 0) / d
v  = E * b / C
p  = E
```

Zero disagreement gives zero gain/harm; unknown future support is never a known
negative label. Fitting excludes incomplete outcomes, matches the old draw counts,
and uses the same predictor-specific causal features and source preprocessing.
Forest leaf averaging preserves `u+ + u- <= p` and `v <= p`.

At inference, only causal features and the frozen forecast disagreement are used:

```text
positive risk = d * predicted(u+)
net risk      = d * (predicted(u+) - predicted(u-))
denominator   = C * predicted(v)
```

The old learned net-gain score is the objective, not an observed oracle gain.
The two population controls maximize this same objective on this same eligible
set, subject to their respective risk sum being at most `.02 * sum(denominator)`
over forecastable targets at one recording/frame. Net-matched-positive adds the
exact intervention count of the positive-population control. It separates count
expansion from which targets are selected. Signed risk can be negative, so an
individually over-budget target cannot be discarded before the query is solved.

## Interpretation

The net constraint is weaker: it credits predicted easy improvements against
predicted easy harm. This can improve the net-degradation tradeoff without making
any individual's prediction safer. It is not legitimate to describe improved
accuracy under this constraint as a better positive-harm guarantee.

With exact conditional moments, a causal decision and matching outcome support,
the expectation of the selected easy excess cost would equal the selected sum
of conditional net moments. Those assumptions are not established here. The
fitted moments are estimates, their training labels require complete support,
some forecastable targets have missing future labels, and site shift remains.
The primary available-point ADE metric and the complete-label training estimand
are not identical. Numerical feasibility only checks predicted coefficients.

Accordingly, all policies retain empirical easy degradation, zero-CV harm,
worst-site/seed results, incomplete-label selections and partial-label gain bounds.
There is no risk certificate, independent calibration or deployment selection.
Four design-exposed sites and three seeds support exploratory paired contrasts;
overlapping windows are not independent samples and site bootstraps are limited.

This isolates a target/accounting mechanism. There is no geometry term in this
experiment, so it cannot reverse the earlier negative nonadditive-interaction
finding or establish a neural forecasting or world-model contribution by itself.
