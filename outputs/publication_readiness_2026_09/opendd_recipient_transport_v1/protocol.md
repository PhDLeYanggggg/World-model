# OpenDD development: who should receive a fixed trajectory edit?

This experiment follows the Songdo fixed-recipient contrast. It uses the
already exposed vehicle-only OpenDD roster of the frozen additive pair
study. At each 8-observation/12-prediction query there are two edits. Keep
one edit unchanged, restore the other vehicle to CV, and move its edit to
a previously unedited vehicle. Express the correction in the old vehicle's
heading and apply it in the new vehicle's heading. Both headings use the
last three observed displacements and must cover at least 0.15 native
coordinate units. The edit keeps its norm at every prediction step. The
incumbent, query, edit count and other path are fixed across comparisons.

The target is the direct conservative ADE improvement over the incumbent
for the two changed trajectories, divided by their summed change budget.
On unobserved steps the difference is bounded by minus the norm of the
trajectory change. Training consumes actual outcomes only on fitting
sites. Features use history, observed shared-frame spacing, neighbor
descriptors, and the frozen edit shape; no target or target availability.

Use the three original seeds and seven held sites. For each held site H,
each training site's proposal comes from the archived model excluding H
and that site. Fit two identical small networks for 300 steps with the
same initialisation and 32 candidate draws per donor per step. One uses
mean donor loss; the other uses a log-sum-exp of donor losses (temperature
0.05). Both measure Huber loss relative to the zero-gain predictor.
Keep the original recipients unless the best proposed swap has predicted
positive gain. Zero is fixed, not chosen on held outcomes. Compare both
heads with the archived two-edit additive incumbent, and each other.

Score the existing repeated pair directions, first averaging directions
within seed and location, then seeds within location. Report a paired
3,000-draw location bootstrap for direct conservative ADE gain, available
gain, conservative relative trajectory gain, and dual-pass rate. A
useful result must beat the incumbent on conservative trajectory gain
or dual pass without evidence of worsening the other, and must not be
claimed as a new test or deployed. The original recording-level 2% risk
and both reference checks remain unchanged. Undefined support is not a
pass. The exploratory hindsight count is not a deployable result.

No new external outcomes, no altered release policy, no future features,
Stage5C or SMC. Native dataset units and annotation steps only.
