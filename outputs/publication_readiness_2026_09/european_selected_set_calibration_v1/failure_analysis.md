# Two Different Failures Remain

This is posthoc source-OOF accounting, not a learned taxonomy or an inference
gate. No future-availability, known-row count, outcome or case ID is introduced
as a feature. The original risk budget and scientific comparisons are unchanged.

## Unknown-Outcome Support: Three Views

The controller1/source067 heads17,29,43 retain respectively6,10,10 occurrences.
Their known-label easy positive-harm ratios are0.8591%,1.9023%,1.0624%, below2%.
But1,2,1 selected unknown outcomes contribute causal-disagreement envelope mass,
raising their finite-completion upper ratios to13.8106%,28.0208%,9.5028%.
These are not observed13%-28% errors. Nor may the unknown outcomes be assumed
harmless so the gate passes.

The unknown envelope mass remains0.633499,2.000526,0.633499 in the three new
selected sets, while known easy reference mass is only4.891321,7.659401,7.505595.
Dropping more rows did not remove the unsupported exposure in proportion to its
denominator. All three conservative net-utility lower masses are negative.
This distinguishes insufficient completion support from measured harm, without
claiming missingness is random or recoverable from past inputs.

## Genuine Observed-Harm Failure: One View

Controller0/source008/head43 was supported under the parent calibration. It
shrinks from3180 selected occurrences to6, with no unknown labels retained.
Its observed easy positive-harm ratio rises from0.350692% to6.915766%; the new
completion ratio is identical to that observed ratio. Net gain becomes negative
(-0.355272 in dataset-local aggregate error units).

Only0.156266% of the parent's known easy reference mass remains, whereas3.081613%
of its easy harm remains. This is direct accounting evidence that monotonically
shrinking a decision set does not monotonically lower its realized harm ratio.
It is not explained by missing labels, and it is not a new mathematical theorem.
Whole-easy degradation remains small, illustrating why net easy preservation
cannot replace the selected-positive-harm criterion.

## An Apparent Rescue With Very Little Utility

The same source/controller with head29 gains complete support, but selection
shrinks4827->27 and conservative utility1657.891277->8.299311. It retains only
0.500594% of the former utility lower mass. This one rescued view does not offset
the lost supported head43 view or the negative primary utility contrast.

## What the Control Rules Out

- The implementation did not silently reuse held-recording labels for that
  recording's coefficients; held-source exclusion and exact action hashes were
  checked independently.
- This is not a stopped optimization: no calibrator hit the8-round cap.
- Harm-only OOF actions are identical to the parent. Repeating this global
  correction is not a demonstrated way to learn conditional harm.
- Making global calibration progressively conservative is insufficient here:
  both missing-label support and a fully observed harmful retained set survive.
- The hypothesis is rejected for this fixed empirical90th-percentile monotone
  algorithm. It does not prove that all selection-aware calibration is impossible.

## Next Discriminating Work

Do not fit the four failed views or run target transfer for this rejected arm.
First compare source-recording decision stability with existing causal-support
and cross-head controls: does the same past-only decision change radically when
one calibration recording is removed, and does that information add anything
to the already tested support features? This question can be evaluated on the
fixed source packets without opening independent roles. Any new learned head
or fallback rule needs its own source-only frozen comparison before outcomes.

For unknown-label cases, distinguish genuinely unavailable labels from existing
partial-label tracking support. Do not silently relabel or change the evaluation
mask. For the fully observed case, focus on conditional harm/ranking stability,
not merely increasing an overall reference margin. These are evidence-directed
next questions, not claims that the next repair will work.

No transfer, deployment, Stage5C or SMC execution. Pixel/raw-frame,
detector-silver development evidence only.
