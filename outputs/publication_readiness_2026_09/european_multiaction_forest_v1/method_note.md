# Sample-Matched Conventional Continuous-Cost Control

## Supervised Task And Information

The forecasting producer and physical action paths are frozen. This study fits
a decision-cost estimator, not a new trajectory forecasting model. Each query
has up to seven causally available candidate actions: the six preceding
physical candidates and the frozen neural forecast. Analytic no-op actions
remain unavailable for genuine intervention. Retaining the floor is not an
eighth learned candidate label.

For each available action the estimator predicts benefit B, positive harm H,
floor reference R, easy-reference ER, and easy-harm EH. These are supervised
future-cost targets on fitting data, not inputs at inference. Unknown future
outcomes are excluded from supervised fitting draws and retained in readout.

The causal inputs are the parent's 380 history/rollout descriptors and 18 static
image-region descriptors, seven action-identity coordinates, and
log1p(envelope / frozen query scale). The neural comparator also receives the
causal envelope in its decoder. The static polygons are camera-region context,
not verified walkable maps or metric geometry. Neither model receives recording
IDs, a future-support mask, ground-truth endpoint or evaluation-only easy label.

## Matching The Finite Fitting Measure

Let c_i be the exact number of times query i appears in the original source-
balanced 2,000 x 128 draws. For the active bank let A_i contain its legally real
actions, m_i = |A_i|, and r_j be the shared fitting-only target RMS. The standard
symmetric five-moment component is proportional to

```text
sum_i c_i / max(m_i, 1) * sum_{k in A_i} mean_j(
    prediction_ikj / r_j - target_ikj / r_j
)^2.
```

Expand each positively sampled (query, real action) once, with weight c_i/m_i.
A common rescaling of all positive weights does not alter this weighted
quadratic minimizer. A query with no real bank action contributes zero, as in
the original masked loss. This reproduces the finite fitting measure for the
symmetric component without storing 256,000 repeated windows per model.

The tree input and target matrices are current-bank working matrices, not
materialized trajectory episodes. Report their size, unique sampled queries,
expanded candidate count and exact feature/target/weight/draw hashes. Minimum
leaf size applies to these expanded unique candidate rows, not to a literal
table containing all duplicate query draws.

## Important Non-Equivalences

The neural comparator uses a fourfold asymmetric underestimated-harm penalty,
centered within-query ranking, a constrained decoder and gradient optimization.
The forest uses standard multi-output squared-error leaves and random split
search, then legal moment projection. It is not a one-factor architecture or
equal-compute ablation. Matching information, normalizers and query weights
does not prove that an observed difference is caused uniquely by architecture.

Both approaches ultimately retain nonnegative moments satisfying B+H<=D,
B<=R, ER<=R and EH<=H, and use the unchanged causal positive-gain and predicted
all/easy 2% intervention gate. The original 1x gate is primary; fixed 2x/4x
harm sensitivities are reported, never selected from exposed outcomes.

The inherited aggregate calibration MSE reports all seven actions, including
inactive-bank predictions, to preserve the parent's diagnostic definition.
It is not an active-bank deployment-error estimand. The separate fitting-region
mechanism check restricts its all-legal pools and selected regions to that bank.

## Comparison And Claims

Primary paired differences are forest minus frozen static-scene neural cost
heads, evaluated in each bank on the same full query roster. The mixed-minus-
physical contrast asks whether access to the frozen neural candidate helps the
forest-controlled policy; it does not train or ablate the forecasting producer.
Known utility, unknown-completion bounds and required-recording risk are
separate results. Matching intervention counts compares utility, not safety.

Completion contrasts use fixed observed floor-error mass as their normalizer.
They are not percentages of a fully observed population error denominator.
Hard/easy pooled known-error summaries are supplementary and cannot replace
the locality-balanced primary estimand. Zero subgroup reference is not_estimable.

All SOURCE inner-held views have already been exposed during development.
Three controller seeds and locality bootstrap are conditional evidence, not
three independently trained forecasting pipelines or new confirmation data.
No deployment promotion follows from an average gain alone.
