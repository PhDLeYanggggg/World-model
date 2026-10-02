# Fixed-Routing Geometric Leaf Refit

## Material Passport

Mode: experiment execution. Population: previously exposed European source
development data, obs8/pred12 raw stride12, image-local detector-silver.
Inputs: cached-verified source forests and original train/validation partitions.
Result: new fitted terminal values and source-held paired evaluation only.
Not independent confirmation, physical safety, metric, seconds, human gold,
true3D, foundation or CVPR-ready evidence. Stage5C/SMC off.

## Hypothesis and Prior Evidence

The previous decoder control confirmed that joint benefit/harm projection can
create optimistic risk forecasts, but preserving harm did not repair support.
This experiment moves upstream: original leaves average absolute costs from
training samples with different causal disagreement envelopes. Test whether
averaging envelope-relative costs inside the same learned routing improves the
conditional-cost estimator before projection.

This is not the earlier neural fractional-BCE auxiliary. That experiment changed
a shared network and degraded reference prediction; its negative result is
retained. Here reference moments, features, tree splits, training membership,
query weights, support and thresholds are frozen. No new tree partitions or
neural parameters are learned. The only newly fitted quantities are three
terminal values per existing leaf. Do not call this full neural retraining.

## Exact Intervention

For each existing tree and its known training rows, recompute weighted leaf
means of B/e, H/e and EH/e, with e the past-only predictor-disagreement envelope.
Use the original query-balanced sample weights. For zero e, require exact zero
B,H,EH and define their fractions as zero. Reject envelope violations beyond
1e-5 relative tolerance. Record any numerical-roundoff fraction clipping.

At inference, multiply mean fractions by the query's causal envelope. Average
trees, clip B<=R, keep original R and ER exactly, and retain EH<=H. Convex
averaging ensures B+H<=e without the original proportional compression.
No validation/held labels or outcome availability enter inference.

Verify original eight-output training fingerprints. Reconstruct original raw
five-moment values in every populated leaf from the exact training rows and
weights; compare to the frozen tree values. Record leaf envelope mean/std and
their source-query mismatch, including selected, added, removed and observed
unsafe slices. Slice labels are offline diagnostics only.

## Paired Evaluation and Decision

All 72 heads: upstream43, head seeds17/29/43 across original12 source localities.
No favorable locality, seed, threshold or target scale is selected. Parent
prediction/decision hashes must match previous receipts.

Primary diagnostic: refit minus original source-held signed-score MSE, using
the original training-only normalization and query weights. Also compare
conservative utility, selected harm/reference and unknown-outcome completion
bounds at the unchanged2% budget. Known partial labels stay as defined; wholly
missing rows remain unknown. Empty/undefined action sets are not safety passes.

Match intervention counts within each recording+frame query using the smaller
count, ranking each policy by its own predicted utility. This is an actual
deterministic paired subset comparator, unlike the prior expected-random-thinning
diagnostic. Report full and matched utility changes normalized by full known
reference mass, not as ADE/FDE improvement.

3000 locality bootstrap draws, seed43; average heads/controllers within locality
first. These are nominal exploratory intervals, not independent end-to-end
replicates or confirmation after repeated development experiments.

Advance to a separately frozen directional control only if signed-MSE interval
is wholly negative, both full and matched utility intervals wholly positive,
complete source support does not decrease, risk-violation count does not rise,
and worst easy completion-upper risk does not rise. Passing this preliminary
gate is not deployment approval or satisfaction of the2% risk budget everywhere.
Do not adapt thresholds, invoke old residual calibrations or open independent
roles after a failed result.

## Execution and Recovery

Local arm64 CPU4/interop1/workers0; no MPS or resource probing. Pilot one fixed
first head, then complete all72 if resources permit; no slow-run scope reduction.
Per-head checkpoint, heartbeat and exact repeat of fitting/inference/readout.
Resume reconstructs and verifies existing groups without overwriting differences.
12h bound. Original source models remain immutable.

Local disk is below the unchanged10GiB numerical-cache reserve. New fitted
terminal arrays are sent directly from memory to the authorized owned CREATE
directory `/users/k24101830/m3w/european_leaf_geometry_v1/inputs/`, capped at128MiB.
This remote process performs file I/O/hash verification only, never training on
a login node. All fitting is local. No new scheduler job; inspect live M3W jobs
before starting. Personal remote quota is unknown; write failure is a hard
storage error, not permission to delete other artifacts. Local output is limited
to8MiB of aggregate reports. No raw rows, images or model arrays enter Git.
