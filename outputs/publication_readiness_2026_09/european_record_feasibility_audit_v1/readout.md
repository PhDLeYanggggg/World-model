# Frozen TRAIN Diagnosis: Local Conflict and Incoherent Reference Estimates

## Material Passport

Fresh frozen-state calculation,not training. PID76259exit0 evaluates36fixed
physical/mixed states in74.935765s,peakRSS5,850,284,032B. Replay PID76817exit0
reproduces every result exactly in73.474452s. No optimizer update,held score,
independent-role opening,threshold selection or deployment change occurred.

## Findings

The mixed bank has18frozen states. Only2trigger global gradient clipping at
their final weights. Four have support-versus-risk gradient cosine below-0.9;
five have support gradient norm above1e-7. Some remaining positive-support
violations have saturated near-zero gradient despite capped duals. This is
localized constraint conflict/dead-gradient evidence,not a universal clipping
failure or proof that causal features cannot distinguish the required rows.

Across repeated mixed recording/head views,125all-risk constraints are
positive;51have multiplier below1. This is an optimizer-scale observation,
not a proof that multiplier1is necessary or sufficient. The audit uses the
final weights and final duals,not the preceding training monitor.

No mixed-bank record with deeply feasible support(excess<-1)has a positive
support dual. Thus excessive pressure on already satisfied support is not
supported as a dominant final-state explanation. A generic augmented-Lagrangian
change is not the first repair selected from this diagnosis. Non-convex and
non-differentiable constrained learning has established prior work; this
diagnosis does not establish a new optimization theorem or method.

The supervised R and ER targets are exactly identical across candidates for
each known row. Their frozen estimates are not: for the mixed bank,the
conventional median across source/context/head views of each view's median
candidate-relative spread is0.393994for R and0.558204for ER. Relative spread
is(max-min)/mean across all seven candidates for a row. These repeated views
are not independent samples,and the statistic is not a realized error ratio.

The largest compressed-feature clipping fraction is0.353480,but it is in the
causal envelope channel of a small source. Reference channels are not broadly
being clipped. Candidate-specific reference estimates are instead a concrete
structural mismatch worth a matched test. Enforcing coherence need not improve
predictive calibration or safety; different candidate features can carry
different information about the same target.

## Next Controlled Experiment

Keep parameters,initialization,loss,optimizer,draws,forecasts,data and the2%
decision/evaluation budget fixed. Estimate R/ER once from the existing causal
floor feature token,using the same network with zero action code;retain
candidate-specific B/H/EH. Compare the resulting full mixed policy against
both physical-only and the unshared parent on the same registered roster.
Require real recovery and full disabled-sharing neutral controls before54fits.

This tests candidate-reference coherence before adding feature capacity or
changing several optimizer factors together. It is not a claim that shared
heads are novel or that correcting an invariant guarantees safe deployment.

Prior work consulted,not reproduced by this diagnostic:
[Cotter etal.,JMLR2019](https://www.jmlr.org/papers/v20/18-616.html),
[Cotter etal.,ICML2019](https://proceedings.mlr.press/v97/cotter19b.html).
Independent calibration/confirmation remain unopened. M3W stays a2.5D
trajectory research scaffold;no metric/seconds,true3D,foundation,Stage5C orSMC claim.
