# Why This Is Not A Reference-Support Repair

## What The Experiment Establishes

The C-selected forest reduces normalized ER prediction MSE relative to a
constant prior, but the existing learned neural reference is better on both
fixed targets. Selected-minus-neural MSE reductions are -1.42915e-4
[-2.22749e-4,-7.55963e-5] conditional on known labels, and -1.43482e-4
[-2.21417e-4,-7.79073e-5] on original bounded labels. Twenty of23supported
source groups favor the old output under either target. All are retained.
The two-part head loses to direct regression and event Brier lift is unproven.

These are exact same-row output comparisons with3000group resamples after
producer/seed averaging, not independent confirmation. Selection remains
16direct/2hurdle from internal Csource folds, with no Hreselection.

## What It Does Not Establish

This is not a matched architecture ablation. New heads use382raw descriptors:
local380at the full-proposal candidate, logcurrentquerysize and logcausal scale.
The inherited neural CostHead encodes all seven actions and pools current-query
hidden states before producing a candidate-invariant R/ER. Its local-history
decoder zeros the48shared-motion slots, but still has learned query-context
pooling. It also uses its C-fitted preprocessing and a different multi-moment
bounded objective. The new head excludes unknown labels from supervised loss.

Input aggregation, learned capacity and target differences are plausible
contributors. This comparison does not isolate their effects and does not
prove forests, hurdle models or conditional ER targets intrinsically fail.
In particular, the original neural reference already shares R/ER across
candidates; sharing cannot be advertised as a new structural repair.

## The Remaining Scientific Failure

The fixed-decision diagnostic found51active shared-arm recording views with
no observed selected easy-reference mass;47have some supplied future label
for every selected row. Positive mean predictions do not ensure realized
reference support in a finite selected set. A marginal event classifier also
does not establish a record-level support probability. No independence of
agents or missing-at-random assumption is justified here.

No head in this study changes a trajectory, decision, calibration threshold,
source role or the original2%contract. The earlier63defined all-risk
false-feasible views are not repaired. Original24primary stays not_estimable;
23supported sources remain exposed development. No deployment upgrade.

## Next Controlled Question

Retain the old neural reference. First distinguish reference amount prediction
from finite selected-record support using the fixed original risk contract.
A follow-up input/aggregation test must match targets, weighting and budget,
fit all preprocessing within the existing inner source folds, and compare
floor-specific plus raw current-query context with the current row descriptor.
Do not inject the globally C-trained critic into inner validation features.
Freeze any proposed admission rule on Cbefore another exposed Hreadout; do
not call that independent confirmation or change official/reserved roles.

Only an actual policy evaluation retaining all unsupported and empty views
could support a risk-repair claim. Neither a lower prediction loss nor fewer
undefined denominators alone is sufficient. Main v9 PDF was not recompiled.
No metric/seconds/physical-safety/true3D/foundation/human-gold claim, Stage5C
execution or SMC. Private checkpoints and row arrays are not Git artifacts.
