# Random Seeds Do Not Explain Away the Risk Failures

Result source: fresh_run for48 new heads and the frozen readout; cached_verified
for24 original43 heads. This is an exposed-development diagnostic, not an
independent test. The original2% selected-reference risk budget is unchanged.
No threshold is tuned, no favorable seed is selected and deployment is unchanged.

## What the Crossed Control Isolates

The earlier six screened easy-risk failures all involved shared seed43. That
seed affected the upstream forecaster/controller as well as the cost head.
Here the upstream43 predictions, source partitions, optimization features,
targets, weights, masks, preprocessing and128-tree settings are fixed. Only
cost-head seeds17/29/43 differ. The same failures still occur, so an explanation
based only on an unlucky cost-head seed is inadequate. This does not prove that
one particular upstream component is responsible.

All three heads have30 defined screened risk directions and42 undefined ones.
Easy-risk violations are7,5,6 respectively. Changing a seed alters the selected
population and can move individual directions across the threshold, but no
seed satisfies the risk requirement. Equal source-pass counts11/24 do not mean
identical accepted sources: heads17 and29 differ in two source groups.

## The Six Motivating Failures

The following are selected because they failed in the parent study; they are
not an unbiased estimate of failure prevalence. Values are screened easy
positive-harm/reference percentages, with a2% budget.

| Upstream context; source -> target | Head17 | Head29 | Head43 |
|---|---:|---:|---:|
| single0/controller2;007 ->119 |3.4132|3.4132|4.9827|
| single1/controller0;067 ->020 |2.1042|undefined: source fallback|2.1042|
| single1/controller0;048 ->067 |2.2457|1.9323|2.1478|
| single1/controller2;074 ->112 |1.8907|2.6020|7.5506|
| single1/controller2;126 ->112 |1.7931|2.0583|2.1020|
| single2/controller0;007 ->124 |4.5322|4.5322|4.5322|

Three of six fail for all three raw heads; two of six fail for all three
source-screened heads. In067 ->020, head29 is screened out: its raw risk is
still2.1042%, so this is abstention, not a predictive repair. Heads17 and29 each
retain four of the six parent screened failures, and introduce three and one
additional screened failures respectively. Reporting only the improved cases
would hide those tradeoffs.

## Two Different Miscalibration Mechanisms

For known selected rows define easy harm H and easy reference R. The offline
budget-excess error is exactly

`(H - .02 R) - (predicted_H - .02 predicted_R)`

`= (H - predicted_H) + .02 (predicted_R - R)`.

This is an accounting identity, not a causal intervention or inference feature.
Unknown selected outcomes remain separate. The numbers below are sums in the
native image-local error units of each direction. Compare the two columns
within a row; do not pool their magnitudes across differently sized directions.

| Parent head43 direction | Harm-error contribution | Reference-error contribution | Larger positive contribution |
|---|---:|---:|---|
| 007 ->119 |+0.008430|+0.042656|reference overestimate|
| 067 ->020 |+0.000607|+0.028975|reference overestimate|
| 048 ->067 |+2.085779|-0.708989|harm underestimate|
| 074 ->112 |+0.426801|-0.016436|harm underestimate|
| 126 ->112 |+3.055098|-0.878763|harm underestimate|
| 007 ->124 |-0.003315|+0.010005|reference overestimate|

The last direction is particularly informative. All heads predict an easy
reference sum around0.68-0.70 while the observed sum is0.1956. Head43 actually
overpredicts harm, not underpredicts it, yet predicts a risk of1.7503% while
observed risk is4.5322%. Penalizing harm alone does not address this denominator
error. Conversely,074 ->112 is dominated by underestimated harm, so lowering
reference predictions alone is not a complete explanation either.

Among all screened violations, the larger positive error contribution is
reference overestimation in6/7 for head17,3/5 for29 and3/6 for43. This descriptive
classification does not identify a unique cause. Four parent failures have no
selected missing outcomes; missing labels alone cannot explain their failure.

## What This Does Not Establish

Whole-population easy ADE can remain nearly unchanged while selected-tail
risk fails. Source-only finite-completion support is not cross-domain safety.
Undefined views are not passes. Different heads do not create independent
upstream training replicates, and12 exposed localities do not become a fresh
confirmation set. A small mean gain does not justify deploying a failing head.

No metric, seconds, human-gold, true3D, foundation or physical-safety claim.
These are obs8/pred12, stride12 raw-frame, image-local detector-silver results.
Stage5C execution and SMC remain off.

## Next Falsifiable Repair

Keep all three heads and the frozen floor. Before changing another loss or
threshold, register a source-recording-only, cross-fitted component-calibration
control: estimate harm upper and reference lower support separately, include
their coverage/abstention cost, and test both against the unchanged2% criterion.
Include a no-calibration comparator and matched intervention counts. Select no
calibration rule using transferred outcomes. Check source analogues for both
error mechanisms rather than fitting only the six motivating failures.

This proposed repair is not_run. If source-only component support cannot
transport, the next issue is conditional support/data diversity, not permission
to loosen the risk budget or open independent confirmation prematurely.
