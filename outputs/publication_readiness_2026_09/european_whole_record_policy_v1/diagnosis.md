# Whole-Record Diagnostic: Sampling Often Misses Constraint Violations

## Material Passport

Fresh_run frozen-state calculation on original TRAIN fit recordings only.
18 parent hard_record checkpoints are cached_verified. No optimizer update,
new threshold,held scoring,independent-role access or deployment change.
Diagnostic PID184 completed with exit0; all18 fixed states retained.

## Evidence

277 recording/risk-channel occurrences have positive exact empirical excess.
In256 of them,at least half of256 repeated64-row minibatches have nonpositive
estimated excess. Across the277 positive cases,the median nonpositive-draw
fraction is0.81640625. The exact independent-draw probability of missing every
positive contributor has median0.74014326.

At the same frozen parameters,16 sampled loss-gradient directions per state
are compared with the exact full empirical gradient.94/288 have negative cosine;
median cosine0.18333681. No model update is made during this diagnostic.

These are repeated fitted states and channels,not277 independent recordings or
288 independent experiments. The figures do not identify a universal cause of
all held failures. A noisy minibatch estimate can be unbiased in expectation
while frequently appearing feasible at a finite batch size.

## Consequence for the Registered Experiment

Proceed with exact_dual and exact_gradient,keeping the original stochastic
policy as a frozen comparator. Exact_dual isolates the full-record constraint
update while retaining the original primal sample stream. Exact_gradient
also replaces the primal estimator; extra computation is explicitly counted.
Both preserve the original2% outcome budget,model and final logit threshold.

This diagnosis supports testing the estimator hypothesis; it does not establish
that whole-record optimization will improve utility or transfer. All final
weights must be frozen before the existing exploratory held-recording readout.
No better secondary arm will replace the registered primary after readout.

obs8/pred12 rawstride12,pixel/local detector-silver. No metric/seconds,physical
safety,human gold,true3D,foundation or submission-ready claim. Stage5C/SMC off.
