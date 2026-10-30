# Fixed-bank intervention evidence for the manuscript

This is an exposed source-development results component, not the paper's
independent main experiment or a claim of submission readiness. The complete
readout and separate replay receipts must accompany it. Anonymous manuscript
integration must not include author-profile, private data or account metadata.

## Experimental question

We ask whether a learned history forecast retains value when competing against
a reoptimized physical action bank, and whether current-query allocation
preferentially improves the neural bank. The forecast producer,18 cost heads,
causal inputs, action eligibility and all/easy2% predicted-budget rules are
fixed. Four controls share the physical independent rule's intervention count
in every recording/frame query, including zero-count queries. Physical and
expanded joint policies are both optimized; deleting neural uses is not used
as the physical comparator.

## Main fitting observations

|Bank and decoder|Known ADE gain over frozen floor (%)|
|---|---:|
|Physical independent|1.710155|
|Expanded independent, trimmed to the common query count|1.900218|
|Physical joint|2.322960|
|Expanded joint|2.544091|

The expanded-minus-physical joint contrast is+0.221131 percentage points
(95% locality interval[+0.065452,+0.430560]). Accounting for wholly unknown
rows using a paired forecast-disagreement bound gives a completion-lower
contrast of+0.154613pp[+0.005466,+0.362079]. This is a modest retained neural
contribution after physical reoptimization, not an absolute statement about
the best possible physical predictor.

Joint allocation improves the physical and expanded banks by+0.612804 and
+0.643873 known pp, respectively. However, their four-policy interaction
contrast is+0.031068pp[-0.003349,+0.071656], with completion-lower contrast
-0.012419pp[-0.061671,+0.027880]. The experiment does not establish a special
neural/joint synergy. This negative interaction result limits the method claim.

## Safety and scope

The expanded joint policy supports47/444 required recording-risk views and
has322 known selected easy-risk violations. It fails the unchanged deployment
requirement despite favorable mean utility. Only6/8 fitting checks pass; the
two failures are original complete-recording risk and all-query optimality.
Four physical and seven expanded solver outputs fail numerical postchecks and
retain checked feasible anchors. Neither small numerical fallback counts nor
a positive ADE interval explains away the observed risk failures.

The74 recordings and12 localities contribute149,510 unique agent-target rows
and18,766 current-frame queries, reused across contexts and three cost-head
seeds. Bootstrap draws resample localities
after averaging repeated views. The intervals describe conditional exploratory
variation, not independent or multiplicity-adjusted confirmation. The single
forecaster seed remains fixed. SOURCE resubstitution cannot establish unseen
generalization or justify calling this a new successful world-dynamics model.

## Reporting requirements for any later main result

Keep the same query/cardinality controls, original selected positive-harm ratios,
unknown outcomes, easy-support status and full-policy recording requirement.
Report all source slices, tails and forecast-pair geometry proxies from the
aggregate archive; do not treat detector-silver geometry as physical safety.
The observation protocol is8 past/12 requested future steps at raw stride12,
not historical raw-frame t+50. Full-roster FDE remains not estimable when a
required endpoint is missing. Coordinates and time are not metric-calibrated.

An eventual paper still needs a useful risk mechanism, exposure-compliant
independent evaluation, appropriate public strong methods, whole-pipeline
reproducibility and complete evidence-linked manuscript/supplement. No Stage5C
execution, SMC, true3D, foundation-model or human-gold claim is supported here.
