# Query-score units and protected-output capacity

## Material Passport

fresh_run: complete 72-group read-only training-label diagnostic, PID46712,
89.349 seconds. A separate complete replay agrees on every group and the
canonical summary. cached_verified: registered source inputs, original selected
actions and fitting targets. No new weights, forecasts, choices or thresholds.
This is exposed SOURCE fitting diagnosis, not independent calibration.

## Measured unit mismatch

The old denominator is derived from all legal candidate-action moments, while
the query loss supervises the original selected set's signed harm/reference
score. Across 18 source-context/critic-seed combinations:

|Channel|Old denominator / actual selected-target RMS, min|Median|Max|
|---|---:|---:|---:|
|All-agent|27.882090|48.273834|60.599607|
|Easy-agent|9.569486|17.587900|30.682202|

Both RMS values use their declared fitting units. Actual-target RMS is
source-balanced over nonempty fully-known original selected queries; missing
or empty sets are not zero labels. The actual-target denominator is floored
at 1e-6. Median channel-wise inverse-square coefficient changes would be about
2331 and 310 at fixed errors. They are not measured future gradient ratios,
evidence that a different lambda succeeds, or a risk guarantee.

## Finite-label interpolation limits

The protected decoder requires `0 <= easy_harm <= harm <= envelope - benefit`,
with the parent's anchor-roundoff protection. A weighted Euclidean projection
tests this feasible triangle, not independent marginal clipping. There are
5,256,810 reused known legal candidate views; observed harm exceeds its
protected cap in 6,805 views and easy harm in 4,401.

Of 112,596 reused current-query views, 58,970 original selected sets are fully
known, 3,294 select unknown outcomes and 50,332 are empty. The desired signed
query target, translated back through the frozen reference predictions, is
negative in 5,687 all-channel and 19,698 easy-channel views, above its protected
cap in 1,393 and 233 respectively, and violates the easy<=all ordering in 6,311.
These categories overlap. None is an independent sample count, conditional-mean
misspecification proof, unique causal explanation or observed-safety test.

## Consequence and limits

This motivates a separate one-factor query-unit experiment. It does not change
the completed original critic experiment. Protected-output limitations are
reported, not relaxed after looking at labels. Future labels are supervision
and diagnosis only; none enters the inference features.

The original execution records remain immutable. The registration captured
whether the parent's readout was complete at registration time; current replay
must preserve that historical event value rather than recompute it. A cold
execution adapter is pending, so the completed exact replay is not a claim
that the original entry point works unchanged after the parent completes.

Task: obs8/pred12 at raw stride12, detector-silver/local-image coordinates.
No metric, seconds-level, human-gold, true3D, foundation, independent-success,
deployment, Stage5C execution or SMC claim.
