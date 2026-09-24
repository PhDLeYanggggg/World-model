# Partial-Recording Overlap Screen Before Role Assignment

2026-09-24. Material passport: new raw-input-only screen on the fixed European
Squares archive; no model, outcome scoring, goal construction or calibration.
Purpose: turn the previous complete-track screen into a partial-clip check and
prevent overlapping recordings from being assigned conflicting scientific roles.

The fixed input is the V2 raw audit and the 37 provisional locality-group
manifest. Read every released raw CSV, verify its full parsed-row hash, and
construct frame signatures from sorted class IDs and all four box coordinates.
Tracker IDs and source frame offsets are excluded from the signature so an
export with renumbered tracks or a shifted clip origin can match.

Screen every run of eight consecutive nonempty raw frames. Exact signatures use
float64 source boxes. A second candidate screen uses boxes rounded to integer
pixels. Constant signatures throughout an eight-frame run are reported as
uninformative and excluded from duplicate evidence, not removed from any dataset.
Missing frames break a run. Compare all recording pairs through signatures;
retain within-locality and cross-locality candidates separately. Any match is
a review candidate, not automatic proof of a duplicate or permission to drop rows.

This cannot exclude clips re-encoded with different detections, camera geometry,
speed, frame sampling, or errors greater than the quantization. No images or
videos are downloaded to resolve a candidate in this task. No predictive data
role is assigned merely because there are no exact matches.

One recording at a time, private resumable frame-signature cache, no full ZIP
extraction. Pin code and input identities before execution. A pilot estimates
time and memory, then the same implementation runs the full archive. Preserve
all candidates and failures. A second pass reuses only hash-verified signature
caches for deterministic matching verification; it is not a second raw parsing.

All same-place recordings remain indivisible regardless of screen outcome.
SDD obs8/pred12 stride12, easy-risk limits, closed DroneCrowd confirmation and
existing producer exclusions stay unchanged. No metric, seconds, online sensor
causality, true-3D, foundation, deployment or submission-readiness claim.
