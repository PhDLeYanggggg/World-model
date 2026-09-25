# Method Positioning and the Strong-Motion Control

## Primary-source Reading

Freshly checked on 2026-09-25: Scholler, Aravantinos, Lay and Knoll,
[What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction](https://arxiv.org/abs/1903.07933),
v3, January 2020; the authors identify RA-L publication and ICRA presentation.
Read scope: abstract, formulation, CV definition, experiment setup/comparison,
and history/environment/interaction analysis in the [paper](https://arxiv.org/pdf/1903.07933).

Their deterministic model extrapolates the latest observed displacement. Their
ETH/UCY experiment uses eight observed positions and twelve future positions,
and permits incomplete future tracks. They distinguish deterministic comparison
from best-of-20 sampled prediction, and analyze learned environment bias and
the limited benefit of longer history in the evaluated models. These are
dataset/model-specific findings, not proof that all interactions or histories
are intrinsically unpredictable. Their meters/seconds and reported scores do
not transfer to our image-pixel detector-track protocol.

## What This Experiment Adds, and Does Not Add

The hypothesis is narrower than "neural models should beat CV." It asks whether
neural trajectory candidates outperform a strong deterministic candidate when
both have candidate-specific utility and event-risk estimation, common source
folds, equal fitting budgets and identical abstention constraints. Damping is
not allowed to bypass the easy/zero-reference checks. The neural model is not
credited merely because it received a protective gate that the baseline lacked.

This is a necessary ablation for the proposed research contribution, not a new
architecture, a novelty proof or a reproduction of that paper's benchmark.
The six-model forecast bank and upstream neural fitting are unchanged. Only
the simple candidate's utility/risk heads are trained in this version.

Equal predicted-risk limits do not imply equal realized risks or intervention
counts. Within-candidate joint/unary comparisons separately match actual counts.
The comparison cannot establish physical safety, prospective detector causality,
independent confirmation or a general world model. The source-only selected
cutoffs, nested producer shift and missing future labels remain visible.

## Numerical Constraint

With nonnegative predicted harm C_i, predicted event mass D_i, and risk limit b,
the joint condition is sum(a_i C_i) <= b sum(D_i). If C_i > b sum(D_i), any set
including i is infeasible. Removing i before division leaves the feasible sets
unchanged. This elementary feasibility observation avoids huge coefficients; it
is not a new risk-calibration theorem. Both candidates receive the same repair.

The statistical calibration limitations discussed in the preceding
[event-risk positioning note](../european_conditional_risk_v1/method_positioning.md)
remain. Better predicted moment ratios or an optimal solver do not guarantee
actual easy preservation. The final scientific decision must use the complete
registered source comparisons, followed by genuinely separate roles if the
method warrants advancement.
