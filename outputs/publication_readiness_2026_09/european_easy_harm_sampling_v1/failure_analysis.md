# Failure Analysis

## Exposure Changed, Transport Did Not Recover
The implemented repair did what it was designed to do: roughly half of training
draws now contain positive easy harm, rather than a few per batch. Exact p/q
checks preserve the original expected loss. Checkpoints reproduce the draw
histogram and final sampling RNG. Therefore a missing or ineffective sampler
is not the explanation for the failed risk gate.

The narrow in-training benefit is real but insufficient: H_easy component MSE
decreases in 13/18 full and all 18 motion fits. The other moments degrade, and
the H_easy advantage rarely transports to C. A shared representation, fixed
update budget and a label-mass sampler that is not gradient-variance optimal
are plausible contributors; this experiment does not identify which one is
causal. It does rule out treating more positive exposure alone as a repair.

## Conditional Risk Remains Underestimated
The full corrected-joint rule violates easy positive harm in 21 of 72 dependent
locality views, versus 15 under the uniform mean control. Violations occur at
074 (6), 119 (3), 020 (3), 112 (2), 126 (2), 008 (2), 048 (1), 067 (1), 007 (1).
These are repeated seed/role views, not counts of distinct failed datasets.
Two all-event views also fail. Net easy still passes 18/18, showing why average
easy preservation alone cannot certify the intended constraint.

The locality074 example predicts supported easy harm 307.66 versus 1,742.36
observed. Unknown reference mass is about 1.4%, far too small to explain this
gap by itself. The fixed-action B/C audit also remains underpredicted, so this
is not only an artifact of comparing different selected agent populations.

## Positive Allocation Evidence Is Narrower Than Safe Deployment
Query-budget allocation improves accuracy relative to independent dual vetoes
and a query-count-matched hash rule. Neither comparison proves equal observed
risk or physical interaction consistency. The greedy rule may spend an
underestimated budget on precisely the poorly modeled conditional tail.
Against the old raw-neural rule, improvement is mixed. No setting is promoted
just because its point estimate is favorable.

## Next Repair, Not Another Threshold Sweep
First isolate reference-moment and harm-moment fitting: retain verified
reference estimates while testing a separately optimized harm component,
with an equal-budget uniform continuation control. This is a planned controlled
diagnosis, not a trained result. Before allocating another large experiment,
check full-B learning curves and B-held-locality transport; do not choose a
budget or threshold on C. The purpose is to see whether the observed
multi-output tradeoff can be removed, not assume that separating heads solves
domain shift.

If harm remains underpredicted with reference fitting protected, inspect the
causal information/support in high-harm events and distinguish forecast error
from detector-track artifacts. Any label-quality filtering must be registered
and applied consistently, not remove bad C outcomes after inspection. Fresh
transport and independent calibration remain prerequisites for deployment.

No reserved locality is opened, no tolerance relaxed and no model promoted.
Obs8/pred12 annotation steps at raw stride12, image pixels, detector-derived
labels; no metric, seconds, human-gold, physical-safety, true3D or foundation
claim. Stage5C and SMC stay off.
