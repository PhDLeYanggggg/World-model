# Failure Taxonomy and Next Discriminating Work

## Confirmed Findings

1. Implementation failure, repaired: v1 diagnostic accounting used an undefined
   NumPy alias. A regression reproduced it; v2 fixes only the alias. Neither
   scientific criterion nor numerical estimator changed. The original invocation
   is retained as failed, not merged into a successful runtime total.
2. Objective mismatch contributes under this fixed design: cost weighting greatly
   reduces risk error versus uniform harm fractions. Weighted harm accounts for
   almost all of that reduction; the reference offset adds only 0.87% easy-MSE
   improvement. This does not isolate individual source/envelope weights.
3. Conditional localization remains weak: positive-event SSE rises, false-safe
   counts rise versus uniform neighbors, and only 1.266 of the 95.561 observed
   normalized easy-harm mass is predicted on the realized positive-event cohort.
   These are retrospective diagnostics, not inputs or a false-positive rate.
4. Independent channel fitting is inconsistent: 9,019 primary outputs have
   EH>H. Common positive weighted means of valid harm labels have no such
   violation here. Corrected signed-risk targets lose that order constraint.
5. Nominal versus effective support differs: every query has 32 donors, but one
   donor can carry 99.9935% of the quadratic weight. Effective support can be
   approximately one. This is not an independent sample-size guarantee.
6. Reference distance is not established as sufficient: global cost mean has
   lower MSE than local cost, while shuffled local targets are very poor.
   Beating shuffle is not equivalent to learning useful conditional risk.
7. No original-head win: all 12 source, six context and three head-seed easy-MSE
   averages remain worse. Locality020 remains adverse; locality082 worsens even
   against the uniform-neighbor comparator. No source is dropped.

## Not Identified Yet

The weighting contrast does not uniquely separate envelope scaling, source
frequency and source-scale effects. The leverage extrema do not by themselves
prove the cause of query error. Current frozen feature distance may mix histories
whose future annotation reliability and disagreement scales are not comparable.
Detector-silver labels, upstream training exposure and repeated windows also
limit inference. No claim of irreducible unpredictability is supported.

## Next Action, Not Run

First inspect where effective support collapses relative to the causal envelope
and original model uncertainty, retaining every fixed query and unknown outcome.
Use that diagnosis to define a coupled, support-aware cost fit rather than
independent H/EH offsets. Any constrained repair must retain EH<=H and B+H<=E
by construction and distinguish fitting improvements from safe policy utility.
Weighted harm already satisfies the order constraint yet still loses to the
original, so feasibility alone must not be advertised as solving the problem.

Do not use the observed total-harm ratio as a scaling target, tune query labels,
weaken the risk budget or switch the primary arm to the favorable global mean.
Prior hurdle and cost-Newton controls already show that lower loss and better
event-frequency prediction can fail at policy safety. A new protocol must test
a distinct conditional-support mechanism, not repeat those models under a new name.

No neural deployment, held readout, independent calibration or promotion.
Stage5C/SMC off; pixel/local raw-frame detector-silver only. The research goal
remains active, and the failed fitting gate is not a hard external blocker.
