# A Verified Local Gradient Failure, Not a Generalization Claim

Fresh fit-only autograd completed72/72 views, PID50837 exit0,61.038337s,
peak RSS2,587,475,968 bytes. Predictions and all frozen cohorts exactly replay;
hash/provenance/accounting check passed. No new training in this diagnosis.

The full-range head selects58,707 repeated row occurrences, including1,612
unknown outcomes.54,254 (92.4149%) have exact-zero easy-harm logit Jacobian.
76.590339/79.541375 (96.2899%) of positive easy-harm underprediction lies on
zero-J rows. The same cohort evaluated with isolated odds has33 zero-J rows
and zero underprediction allocated there. This is a genuine same-input
optimization contrast, not a consequence of comparing different own cohorts.

Lower clipping of the easy/all-harm ratio accounts for72.924257 (91.6809%)
of this underprediction; all-harm lower clipping accounts for64.310617
(80.8518%). These subsets overlap. On all eligible rows, the full-range arm
has628,949 zero-J occurrences versus1,271 for odds; positive underprediction
at zero J is3,260.252194 versus13.590045. Source-scale-normalized sums are
not ADE/FDE and repeated occurrences are not independent sample counts.

This identifies a specific local gradient obstruction at frozen final weights.
It does not prove that every failure was caused by clipping, that these rows
can never move through shared parameters, or that removing clipping restores
held performance. The preceding held risk failure remains valid. Stronger
gradient clipping and poor selected-risk support may remain independent problems.

## Matched Repair Triggered

The registered majority-underprediction trigger is met. Test one anchor-exact
C1 rational bounded map with no interior hard clipping, preserving all other
training settings and comparisons. Boundary anchors and floating underflow can
still be flat; do not promise globally nonzero gradients. Do not repeatedly
tune on held recordings. All finals freeze before the fixed readout.

## Boundaries

Original TRAIN only. Whole-file transport includes both internal partitions;
held rows discarded before model forward or label analysis. No independent
selection/calibration/confirmation or outer development opened. No deployment
change. Obs8/pred12 raw stride12; detector-track pixel/dataset-local evidence,
not metric/seconds, physical safety, human gold, true3D or foundation success.
Historical Stage37 remains exploratory. Stage5C and SMC remain disabled.
