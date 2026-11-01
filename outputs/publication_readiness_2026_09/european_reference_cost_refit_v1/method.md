# Reference Learning Under Fixed Forecasts

The study isolates a decision-cost repair, not a new forecasting architecture.
For each agent and available candidate, the inherited network supplies five
nonnegative moments: expected benefit B, harm H, reference R, easy reference ER
and easy harm EH. A physical/neural action is admissible only under its causal
availability mask and the original predicted 2% all/easy cost constraints.

## New Reference Branch

Let h be the inherited candidate encoder and c the inherited selected-query
context. The new branch receives the mean of the six physical-candidate h
vectors, c, log(1 + selected count), and selected count / query size.
A zero-initialized linear layer outputs two row-level log-scale increments.

```text
u_R, u_ER = clip(reference_head(context), -20, 20)
R_new[k]  = R_old[k] * exp(u_R)
ER_new[k] = min(ER_old[k] * exp(u_ER), R_new[k])
```

The branch has 262 parameters. Both arms contain the branch; it is inactive in
the locked arm. Thus total capacity is matched but active capacity is not.
Zero initialization reproduces original native-unit predictions exactly.
The reference increments use causal inputs only. Targets enter the loss and
evaluation, not context or candidate construction.

This is a row-shared scale, not a candidate-invariant reference decoder.
It preserves inherited per-candidate differences and exact zeros. Inherited
benefit <= reference is not structurally enforced. These are material limitations.

## Objective And Sampling

Retain the existing coupled pointwise and selected-query loss and add normalized
reference MSE with weight 1. For the locked arm this term has no reference
gradient. For the learned arm it trains reference estimates toward the original
observed targets, not toward permissive budgets. No manually inflated denominator,
oracle harm zeroing or learned inference label is used.

The SOURCE fitting-only RMS normalizers, preprocessing and causal features are
unchanged. Unknown outcomes are masked from supervised cost loss. They remain in
causal sampling and context; visiting a wholly unknown query is not a claim that
its unobserved outcome supplied a gradient.

Each update draws 16 queries, four from each of four source groups. Within a
source, shuffled cyclic sampling visits every query without replacement before
starting another cycle. The budget is:

```text
updates = max(2000, 4 * ceil(largest_source_query_count / 4))
```

Both arms reset the optimizer, share warm weights and query sequences, and run
the full budget. The final endpoint is fixed before policy outcomes, not selected
using SOURCE policy performance. Smaller sources cycle more often. This keeps
source-balanced weighting; it does not create independent observations.

## Decision And Evaluation

Independent selection uses native expected benefit minus harm under the original
per-row feasibility constraints. Joint allocation optimizes summed native utility
at its own independent intervention count under both predicted 2% query budgets.
Invalid or nonoptimal solves retain the feasible independent parent. Common-count
controls trim to the smaller per-query count and measure utility only.

The primary is learned-minus-locked full joint completion-lower ADE utility.
Original observed references, target roster, missing-outcome bounds and every
recording's 2% all/easy evaluation requirements remain unchanged. A decrease in
reference MSE alone cannot pass the study, and predicted budget satisfaction does
not imply observed recording-risk control.

The 3,000 paired locality bootstrap draws average repeated contexts/cost seeds
before resampling 12 exposed localities. They quantify conditional SOURCE fitting
variation, not independent confirmation. All three cost seeds share forecaster43.
This is not whole-pipeline OOF, multi-forecast-seed evidence or a deployable
world-model claim. Stage5C and SMC stay disabled.

## Additional Nonintervening Diagnostic

A post-registration descriptive diagnostic is collected during the original
frozen-action inference/score verification. It reads predicted costs, causal
availability and frozen causal scales only, not targets. It records benefit
above reference, negative implied candidate ADE, easy-reference/harm ordering,
exact-zero references and candidate-reference spread. Both exact signs and a
1e-12 normalized excess tolerance are reported. No prediction byte, choice,
threshold, endpoint or original gate changes. Such violations can identify a
remaining moment-model limitation;their absence is not a calibration certificate.
