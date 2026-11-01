# Method Component: Separating Routing Utility From Forecast Costs

This is a development-method description,not a completed anonymous paper or a
claim of novelty from combining neural modules. Results belong in `readout.md`.

## Decision Setting

At a causal scene query,each agent has seven fixed candidate forecasts and an
explicit baseline action. Let the frozen critic return five continuous moments
for each candidate:benefit B,positive harm H,reference error R,easy reference
R_e,and easy positive harm H_e. These are learned predictions,not known truths.
Candidate inputs contain past motion and causal rollout diagnostics only.

The native utility is B-H. The previous decision-cost experiment altered cost
learning itself and improved observed utility,but distorted moment calibration
and failed both the robust paired endpoint and full recording safety. A separate
support calculation showed its bounded soft family had a positive lower bound
on expected constraint violations. That finding motivates testing a new routing
family;it is not a theorem that the deterministic allocation problem is infeasible.

## Separate Actor

Freeze forecasts and costs,and let a separate actor return candidate offset a.
Its native routing score is

```text
u_i,k = (B_i,k - H_i,k) + a_i,k * s_i * T
```

Here s is a past-only motion scale and T is the original frozen fitting cost RMS.
The zero-initialized output layer gives the original utility exactly. Soft training
uses a softmax over seven normalized scores and a fixed fallback logit zero.
Unbounded offsets allow arbitrarily small intervention probability;this expands
expressibility but does not guarantee finite-training safety,benefit or calibration.

The two matched arms differ only in their context representation:own candidate
hidden state or the pooled hidden state of the frozen causal selected query.
Both include the candidate state,selected-set metadata and frozen moment diagnostics.
This tests whether the selected query adds useful decision context beyond the
same local representation. It does not test future-conditioned scene information.

## Cost-Aware Training

Known SOURCE outcomes supply expected decision utility and original recording
all/easy positive-harm and easy-degradation constraints. Unknown rows contribute
the original causal pessimistic envelope instead of made-up zero-error labels.
Two further terms match the hard decoder's frozen predicted query budgets:

```text
sum_i,k p_i,k * (H_i,k - 0.02 * R_i,k) <= 0
sum_i,k p_i,k * (H_e_i,k - 0.02 * R_e_i,k) <= 0
```

Whole-record/query passes update nonnegative multipliers every100updates;
minibatches retain the previous equal-source/query importance correction.
Costs receive no gradient. This prevents a routing improvement from being
misdescribed as better calibrated moment regression. The objective is still a
soft surrogate,not exact optimization through a mixed-integer program.

## Hard Allocation And Safety Scope

The deployed candidate policy first forms a feasible independent proposal from
original positive-gain/reference support,positive learned utility and original
predicted all/easy2%constraints. A joint scene-query optimizer can reassign all
seven eligible candidates while preserving that query's independent count.
Nonoptimal or numerically invalid solutions retain the feasible parent. Learned
utility changes the objective;it never relaxes frozen cost budgets.

Feasibility under predicted moments is not actual error safety. Evaluation keeps
every required recording and the original observed/bounded risk definitions.
Matched-count controls isolate utility at a common count but cannot certify
the full deployment policy. Missing labels and empty selected easy support are
reported explicitly. The registered primary is a full-population completion-lower
contrast,not a conditional switched-only mean.

## Contribution That Still Requires Evidence

Free routing,deferral and primal-dual training are established principles. The
testable contribution here is a controlled separation of seven-action routing,
fixed continuous costs,selected-query context and original recording-level risk
under partial outcome support. An independent role-respecting evaluation and
matched component benefit are still needed before making a method claim.
No new world dynamics,metric/seconds prediction,Stage5C execution or SMC is implied.
