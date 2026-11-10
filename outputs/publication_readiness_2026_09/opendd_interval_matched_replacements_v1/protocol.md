# Can interval supervision select better replacements at the same count?

Use the completed OpenDD development experiment, unchanged model
weights, original rotation masks, and original risk and reference
criteria. The intervention being matched here is a second angle
replacing the first, not the initial rotation. For each location,
opposite location and seed, take the smaller of the numbers of
replacements proposed by the two uncalibrated agreement arms. Keep
that many of each arm's highest-scored replacements. Resolve ties in
frozen query order. The quota, ranking and choice of alternate angle
use frozen predictions only. Keep all zero-quota directions in the
location summary.

Compare qualified, risk and strict passes, directly paired conservative
trajectory gains, and both endpoints of the relative-motion effect.
Use the original location/seed aggregation and the same 3,000 location
bootstrap draws. Add a random control which selects the same count
uniformly from each arm's own eligible replacements, keeping its
predicted direction. Its expected qualified count is exact. An
outcome-aware ceiling selects among both alternatives at that same
quota; it is diagnostic only. Replay the original unrestricted counts.

The count match uses the whole held direction's score sequence, not
an online policy or a new calibration rule. It addresses mechanism on
already-exposed development data and cannot supersede any external
confirmation or release decision. No new labels, fits or thresholds.
