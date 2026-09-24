# Baseline-Relative Scene Intervention: Method Draft

This section describes the implemented source experiment, not a validated main
paper claim. Predictor training is real Torch; deployment and independent risk
calibration remain separate, unopened steps.

## Prediction and Relative Costs

At an observed query, let `X` contain only past agent and neighbor coordinates,
their observed masks, and the requested future grid. A fixed causal predictor
produces `b_i(X)`. A source-fitted neural predictor produces `n_i(X)`. The
decision `a_i` is binary and the returned forecast is

```text
p_i(a_i) = (1 - a_i) b_i + a_i n_i.
```

The baseline is selected only on the predictor's fitting localities from
constant position, last-difference velocity, two fixed velocity decays and
four/eight-observation OLS velocity. The neural output is a motion-bounded,
baseline-relative forecast. It is not a latent generative rollout.

For a supported future label and masked native-coordinate ADE `L`, define

```text
G_i = max(L(b_i, Y_i) - L(n_i, Y_i), 0)
H_i = max(L(n_i, Y_i) - L(b_i, Y_i), 0)
L(p_i(a_i), Y_i) - L(b_i, Y_i) = a_i (H_i - G_i).
```

The identity follows because the forecast is a binary choice, not a blended
trajectory. Missing future supervision remains unknown. FDE requires the
requested final endpoint and is not replaced by a last-visible point.

## Learning Before Selection

Cost inputs contain past context, masks and disagreement between the two
forecasts. Native positions are normalized using an observed-context scale;
the logarithm of that scale is retained. Means, variances and loss scales are
fitted on source-fitting localities only. No future labels, future-valid flags
or locality ID are inference features.

The ridge control fits both continuous costs. The neural control uses MSE for
benefit and an asymmetric quadratic objective for harm, with underprediction
weighted fourfold. In a correctly specified population regression, the latter
targets an upper expectile, **not the conditional mean harm**. With finite data
and a neural model, even that population interpretation is not a safety bound.
Its output is therefore a harm-oriented score, not a calibrated probability or
guaranteed upper confidence limit. The solver's legacy `expected_harm` field
name does not strengthen this interpretation.

For outer group A, the cost labels on B come from a predictor fitted on C, and
labels on C come from one fitted on B. Both levels exclude A. The final candidate
on A is fitted on B+C. Three independent seeds are retained. The mismatch
between four-site label producers and the eight-site final producer is explicit;
it must be checked empirically, not assumed harmless.

## Scene-Level Decision

For the `m` complete-history targets in a query, maximize predicted net utility
while penalizing excess forecast overlap:

```text
min_a  - mean_i[a_i (g_hat_i - h_hat_i)]
       + lambda * mean_edges[phi_ij(a_i, a_j)]

subject to mean_i[a_i h_hat_i] <= rho,
           a_i = 0 for unsupported or nonpositive-net-score targets.
```

Costs are normalized by the fitting-only baseline ADE scale. The fixed `rho`
is 0.02 in these units; it is not the population easy-degradation statistic.
The actual 2% positive-easy criterion and exact-zero-CV allowance of zero are
reported separately and are not relaxed by this optimization.

Edges and distance scales use only current positions and observed box widths.
Each pair has four forecast-only overlap costs; subtract its all-baseline cost
and retain nonnegative excess. The binary pair term is decomposed as

```text
phi_ij(a_i,a_j) = phi_10 a_i + phi_01 a_j
               + (phi_11 - phi_10 - phi_01) a_i a_j,
with phi_00 = 0.
```

An integer solver enforces the binary products. Objective/risk rescaling is
numerical only; feasibility and primal/dual agreement are rechecked in original
units. Invalid or nonoptimal solutions fall back. The pair proxy is not a
physical collision probability: the data are detector-track image coordinates.

## Attribution Controls

Compare the same frozen predictors under pointwise, predicted-budget independent,
whole-query uniform and joint decisions. Determine a reference intervention
count from the independent decision before any label is read. At that count,
compare independent, unary geometry and full pair geometry with identical support
and predicted harm cap. Removing only the product coefficient isolates the
nonadditive term. Equal count does not imply equal realized risk.

The joint population is a prespecified 1,152-query, 6,116-target source sample;
the full pointwise population is 318,969 targets. They must not be pooled or
presented as interchangeable runs. The matched nonzero analysis additionally
reports its retained queries and failed solvers. All twelve locality groups
are the declared statistical roster; unsupported or zero-reference percentages
remain undefined rather than disappearing from an aggregate.

## What Would Support the Hypothesis

Useful evidence would require source-excluded neural benefit beyond the strong
baseline, learnable relative costs, and actual prediction improvement from joint
decisions over both matched independent and unary controls. It would then still
need frozen independent model-selection/calibration/confirmation evaluation.
Lower training loss, lower optimized overlap proxy, or one favorable mean is
not sufficient. The learned decision components and existing calibration
literature are distinguished in [method_positioning.md](method_positioning.md).
