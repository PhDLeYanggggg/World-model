# Songdo: match the time profile, not just peak displacement

This is a retrospective sensitivity analysis on the original 160
already-scored confirmation flights. Keep the frozen independent
and joint recipients, edit counts, observed inputs, and candidate
directions. Do not refit or rerank. The original comparison used a
common peak displacement; it did not require equal movement away
from CV at each forecast step.

At each step take the minimum displacement among all four selected
edits (two per arm). Match every edit to that common step radius
without changing its nonzero direction. As a second control, shrink
the four original paths uniformly by the ratio of the new common
peak to the old peak. It shares the new peak and the original time
profile up to scale. If the common profile has no positive step, that
view cannot support a matched two-edit sensitivity comparison.
Do not interpret a fallback as a matched or risk pass.

Freeze transformations before rejoining any previously scored label.
Reproduce the original two-arm results, measure movement retained,
and compare observed/lower/upper trajectory gain and relative motion,
available ADE/FDE and original 2% risk outcomes. Average seeds within
flight and flights within the original 14 source components, with
3,000 source resamples. Retain the same CV denominator, easy/hard
cutoffs, masks, and full recording for risk. A stepwise rescaling is
not a validated motion model or a new release rule. Do not open
calibration flights or independent targets in any other source.
