# Source-Development Data Card

## Roles and Units
The unchanged source rosters contain 318,969 rows: A0 has 116,823 rows
(074,082,112,126), A1 has 23,762 (007,110,119,124), and A2 has 178,384
(008,020,048,067). These are overlapping trajectory windows, not independent
agents or independent statistical samples. The three rosters rotate through
forecaster A, controller B and source-held C roles. The same row can occur in
multiple role/seed views; repeated views must not be summed as unique data.

Observation length is eight annotation steps, prediction length twelve, at raw
stride12. Coordinates are image pixels and labels are detector-derived, not
human gold. Neither effective seconds nor metric geometry is asserted.

## Fitting Boundary
Feature normalization and target scales are fitted on B only. Training uses
only known labels, equal-locality sampling and past-only features plus causal
forecast diagnostics. Future coordinates are supervision/evaluation targets,
never inputs. No central velocity, held-out endpoint goals or C/test-fitted
normalization is introduced. The existing forecast, feature, split and model
lineages are hash-checked before training and replay.

## Readout Boundary
All 576 decision views are frozen and pushed before this round's C readout.
C excludes the current A/B fitted chains but has historical development
exposure. Consequently these are source-development results, not independent
confirmation. Six opened selection localities (087,092,093,103,104,125) are
not evaluated here. Twelve reserved calibration and six confirmation
localities remain closed. The readout reports unknown-label coverage instead
of silently treating unavailable outcomes as successes.

## Statistics and Access
Three seed results are averaged within locality before 3,000 bootstrap draws
of four C localities per ordered source assignment. Overlapping assignments
and multiple unadjusted contrasts limit inference. Model weights, per-row
arrays and raw data remain private; only code, configuration and aggregate
reports/figures are committed. See registration_lock.json and decision_freeze.json
for immutable lineage. Stage5C and SMC remain off.
