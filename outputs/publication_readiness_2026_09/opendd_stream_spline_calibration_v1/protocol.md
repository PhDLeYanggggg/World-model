# Nested streaming-slot calibration on OpenDD

Primary contrast: joint shape choice using streaming calibration slots
versus the prior proxy-calibrated joint choice, at the same outer slots,
paths, model outputs, radius, seeds and grid. Secondary comparisons:
past prefix, independent choice, the shared-rule grouping control, and
gain-only choice. The primary paired utility is the completion lower
bound divided by observed CV error mass. The three original 2% checks,
undefined easy support and both accuracy references are unchanged.

The streaming calibration uses the original controller architecture,
objective, rank menu and chronological count budget. For each calibration
location its count head and scoring ranker exclude the outer and
calibration locations. Each count fitting menu also excludes its own
location from the ranker. The nested spline generators and cost heads
remain frozen and are verified against the preceding run. Outer paths
and calibration settings are frozen before scoring.

Secondary calibration arm: on the same joint grid, use the original
recording scorer on calibration donors, with no label input to the policy.
Discard settings that reduce seed-averaged risk or dual pass rates at
any donor location, or reduce macro conservative CV utility relative to
the unchanged analytic shape. Among feasible settings maximize macro
dual pass rate, then risk pass rate, then conservative gain. Resolve
ties by grid order with fallback first. This rule selects thresholds;
it does not change the held-recording pass definition or certify safety.

Three seeds, 3,000 paired location bootstrap draws after seed averaging.
Report shifts in pass counts, interventions, supported easy tail and
pair geometry; no deployment change, no DroneCrowd confirmation access.
