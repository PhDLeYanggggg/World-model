# Method Position and Risk Meaning

## Material Passport

Predeclared source-development mechanism study, not a new risk-control theorem.
This note reuses the bounded primary-source reading recorded in the preceding
[method-positioning note](../european_source_intervention_v1/method_positioning.md)
and [version-specific calibration review](../risk_subsidy_v1/literature_and_claim_limits.md).
It is not a fresh exhaustive literature search or a claim that every newer
revision has been reviewed.

Selective prediction and learned rejection are established methods. Learning a
switch is not novel by itself. The present task retains all agents through an
explicit CV fallback, and separately measures realized added error on easy and
exact-zero-reference events. Useful task distinctions still need matched strong
controls and independent evidence before becoming a paper contribution.
[SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html).

Conformal Risk Control and Learn then Test already provide calibration tools
under stated sampling, loss and testing assumptions. This experiment does not
run their procedures. Predicting two moments and comparing their ratio to 0.02
is neither a finite-sample correction nor a valid test of a population risk
bound. Source folds share models and have influenced research decisions; a
3,000-resample source-locality bootstrap does not make them independent calibration.
[CRC, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf),
[LTT](https://arxiv.org/abs/2110.01052).

The previously read March2026 version of Conformal Policy Control is especially
relevant to optimized policies relative to a reference. Its calibration and
stability conditions cannot be imported by calling deterministic subset selection
a conformal policy. No sampling-based policy control or SMC is implemented here.
[CPC, inspected v1](https://arxiv.org/html/2603.02196v1).

## Exact Quantity Versus Estimated Quantity

Let `E` be the unchanged positive-easy event defined using a fitting-only cutoff,
`R` be baseline ADE, and `H=max(L_neural-R,0)`. The new head estimates
`D(X)=E[R*1(E)|X]` and `C(X)=E[H*1(E)|X]`. The all-event arm estimates the same
moments without the event mask. Frozen net-utility scores determine preference;
the new moments constrain predicted risk, so utility learning is held fixed.

For a single fixed population and a selection rule using only X, the tower
property gives `E[a(X)*H*1(E)]=E[a(X)*C(X)]` when C is the true conditional
expectation. The implemented C and D are fitted estimates, not true moments.
This identity therefore does not certify the implemented rule. Per-query ratios
also do not directly establish a worst-locality or out-of-domain bound. Positive
harm is stricter than signed excess error but can still be underestimated.

The joint rule can borrow predicted denominator mass from agents not selected.
It can change selected identities as its budget changes. Neither monotone
realized loss nor nonadditive interaction value follows from the formulation.
Exact-count independent/unary controls remain necessary.

Zero-CV events have no positive-easy denominator. They remain a separate
unchanged zero-added-harm check. A source-level absence guard rejects novel
intervention if the training fold has none; a few available events are not proof
of conditional support or population safety. If protection comes only from
abstention and useful prediction disappears, that is a failed utility/safety
tradeoff, not successful risk learning.

## Evidence Required for a Stronger Claim

Retain the common forecasts and causal inputs, exact source draws, strong
fixed controls, every seed and the complete population. Establish event-target
lift separately from support-driven abstention and graph-driven value separately
from risk-budget sharing. Only a frozen credible source policy should proceed
to the separately reserved selection/calibration roles. Independent confirmation
remains closed, and no metric, seconds, physical safety, foundation or submission
readiness claim follows from this experiment.
