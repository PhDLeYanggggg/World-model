# Forecast Substitution: Evaluation Definitions and Reproduction

**Working supplement.** This document explains existing estimands and code;
it introduces no new evaluation rule, outcome readout or positive result.
Elementary norm inequalities are not claimed as a new theorem. The native
moving-drone task observes 8 steps and requests 12 at annotation stride 1 in
image coordinates. This is not verified seconds, meters or physical safety.
The original 2% recording contract, support requirements and undefined-reference
rules remain unchanged. Dual-view outcome evidence is reported separately;
this supplement changes no result or evidence status.

## 1. Available-Label Error Is Not Full-Grid Error

For row $i$ and prefix $k$, let $m_{it}$ indicate whether future position
$y_{it}$ is observed. A forecast $p_i$ is fixed before labels are scored. Put

$$
n_i=\sum_{t=1}^{k}m_{it},\qquad
K=\{i:n_i>0\},\qquad U=\{i:n_i=0\}.
$$

For $i\in K$, the reported available-label error is

$$
a_i(p)=\frac{1}{n_i}\sum_{t=1}^{k}m_{it}\lVert p_{it}-y_{it}\rVert_2.
$$

A row with $0<n_i<k$ is partial but belongs to $K$. It is not a complete
trajectory and its missing steps are not filled. Only $n_i=k$ is complete.
An entirely unobserved row remains in the population, not in the observed
error mean. The audit reports all four counts: rows, known, partial and wholly
unknown, with complete rows separately. Known plus wholly unknown equals rows;
partial is a subset of known, not an extra population.

Within a source, let there be $Q$ queries and $N_q$ eligible agents in query
$q$. The fixed row weight is $w_i=1/(QN_q)$ for an agent in $q$; thus queries
have equal mass before all sources receive equal weight. Label availability
does not define these weights. Overlapping windows are not independent trials.

## 2. Exact Scope of the Registered Paired Completion Envelope

Let $p$ be the new actor, $q$ its matched control, and $b$ causal constant
velocity. Define the observed floor-error mass and the observed paired gain

$$
R_K=\sum_{i\in K}w_i a_i(b),\qquad
\Delta_K=\sum_{i\in K}w_i\{a_i(q)-a_i(p)\}.
$$

The reported known contrast is $100\Delta_K/R_K$ when $R_K>0$ and NA
otherwise. No epsilon replaces a zero denominator. The denominator is the
observed floor-error mass, not an imputed complete-population error.

For an unknown row, define the causal paired disagreement

$$
D_i(p,q)=\max_{1\leq t\leq k}\lVert p_{it}-q_{it}\rVert_2,
\qquad B_U=\sum_{i\in U}w_iD_i(p,q).
$$

For any shared, nonempty future support and any target values on that support,
the reverse triangle inequality gives

$$
|a_i(q)-a_i(p)|\leq D_i(p,q).
$$

This follows pointwise from
$|\lVert q-y\rVert_2-\lVert p-y\rVert_2|\leq\lVert q-p\rVert_2$,
then by averaging over the shared support. Positive row weights preserve the
inequality. Therefore the existing row-support completion contrast lies in

$$
\left[100\frac{\Delta_K-B_U}{R_K},\quad
100\frac{\Delta_K+B_U}{R_K}\right],\qquad R_K>0.
$$

This expression keeps the already observed partial-row loss and its observed
reference denominator fixed. It supplements only wholly unknown row contrasts.
It is not a bound on complete-grid ADE for partial rows, a completed baseline
denominator, actual missing-label error, population risk or a confidence
interval. No smoothness or physical dynamics constraint is assumed. The same
comparison uses disagreement between the two compared policies, not the sum
of their separate disagreements from CV. The latter can double-count uncertainty.

### Table 1. Distinct Evaluation Quantities

| Quantity | Partial rows | Wholly unknown rows | Denominator |
|---|---|---|---|
| Available-label contrast | observed steps only | excluded from observed numerator | observed CV error mass |
| Registered row-support envelope | observed steps unchanged | paired max-disagreement bound | same observed CV mass |
| Separate full-grid gain envelope | every missing step bounded | every requested step bounded | unnormalized gain, no invented CV denominator |

## 3. A Separate Full-Grid Bound, Not a Replacement Metric

Existing code also provides a different, unnormalized complete-grid gain bound.
For a candidate $p$ and baseline $b$, define

$$
G_i^{\mathrm{obs}}=\frac{1}{k}\sum_{t:m_{it}=1}
\{\lVert b_{it}-y_{it}\rVert_2-\lVert p_{it}-y_{it}\rVert_2\},
$$

$$
B_i^{\mathrm{miss}}=\frac{1}{k}\sum_{t:m_{it}=0}
\lVert b_{it}-p_{it}\rVert_2.
$$

The complete-grid ADE gain lies between
$G_i^{\mathrm{obs}}-B_i^{\mathrm{miss}}$ and
$G_i^{\mathrm{obs}}+B_i^{\mathrm{miss}}$.
For unrestricted Euclidean targets these endpoints are attainable: at each
missing step, choosing the baseline position gives the lower endpoint and
choosing the candidate position gives the upper. Observed labels remain fixed.
This sharpness statement does not impose temporal continuity, image bounds,
walkability or physical feasibility; with such restrictions the interval can
be conservative and its endpoints need not be attainable.

The implementation permits a positive past-conditioned scale per row and
multiplies both gain and disagreement by that same scale. No future target
sets the scale. The actor experiments' registered table is not silently
replaced by this separate full-grid quantity.

A simple example shows why the distinction matters. Over three steps, let
the control predict 3 and the new actor 2 on a one-dimensional axis, with only
the first target, 0, observed. Available-label gain is 1. If the two missing
targets are both 100, complete-grid gain is negative one third. The separate
full-grid envelope is negative one third to 1; the available-label contrast
cannot establish positive complete-grid gain.

## 4. Why More Fallback Does Not Certify the Original Ratio

For a deployed selected set $S$, positive harm is
$H_i=[a_i(p)-a_i(b)]_+$. The original empirical all-reference check uses
$\sum_{i\in S}w_iH_i/\sum_{i\in S}w_i a_i(b)$, together with its easy-reference
and signed easy-net checks. These are not signed gain alone; improvement on
one agent cannot cancel another agent's positive harm.

Removing an intervention can lower the numerator yet remove a larger reference
mass, increasing the ratio. For example, two equally weighted rows with reference
errors 1 and 100 and harms 0.03 and 0 have a joint ratio below 2%. Retaining only
the first row produces 3%. This finite example concerns the selected-reference
ratio, not impossibility of safe selection or a new general theorem.

If the selected set or required easy reference is empty, its ratio is not
estimable under the retained protocol. Exact equality of two predictions makes
their disagreement zero, but it does not create reference support or a new
certification exception. Whole-recording baseline equality is not added as an
alternative pass. A matched-intervention-count utility control with no usable
counterfactual support also remains NA, not a safety pass.

## 5. What the Source Bootstrap Does and Does Not Establish

Each actor comparison first balances queries within a source and then averages
all 14 source-component contrasts. The existing exploratory bootstrap draws
14 source indices with replacement, 3,000 times. It does not resample individual
overlapping windows as independent examples. The prefix experiment retains seed
39017; the image-conditioned and camera-frame experiments retain seed 39107.
The portable aggregate archive preserves each original index matrix rather
than generating another bootstrap. Quantiles use linear interpolation.

The interval describes variation of the reported source-level statistic under
that resampling scheme. It does not turn globally exposed development sources
into independent confirmation, adjust for project-wide adaptive development,
prove unobserved complete errors, or supply three independent forecaster seeds.
The subsequent dual-view comparison retains its own frozen execution receipt.

## 6. Code and Reproduction Boundary

The actor table is computed by the existing `statistics` function. The separate
full-grid calculation is `partial_gain_bounds`. The companion bindings record
gives both exact source paths and their hashes. The new synthetic tests check
both definitions without opening real trajectory labels. The supplement
builder also records its own source and typesetting hashes. It does not alter
registered model, training, split, feature or metric files.

The portable aggregate package reproduces three completed analyses, not model
training or fresh forecasts. The separate dual-view report records its full
matrix, two complete refits, prediction freeze, scoring and exact replay.
A compile or passing
synthetic test is not evidence of neural improvement, deployment or readiness
to submit. Stage5C execution and SMC remain off.

## LLM Usage Statement

OpenAI Codex assisted the derivations, code checks and drafting. These elementary
arguments and synthetic tests do not replace author verification, real-data
evidence or independent review. The author remains responsible for the claims.
