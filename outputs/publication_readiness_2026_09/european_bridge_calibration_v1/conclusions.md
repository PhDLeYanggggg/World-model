# Aligned Calibration: Transport Still Fails

## What Was Completed
Fresh source-C inference, 72 empirical calibration maps, 288 frozen decision
views and all 18 source-role/seed readouts completed. The 144 scoring models
were cached_verified; no neural weights were updated. Registration 530d16ca
preceded fitting and freeze 34ca0233 was pushed before outcome readout.
There are three seeds, six opened selection localities and 3,000 paired
locality resamples. These are development results, not independent confirmation.

## Main Result
The full neural selected-risk grid passed the registered observed constraints
in every source-C setting (18/18), but in no complete selection setting (0/18).
It reduced easy positive-harm violations from 68 to 52 of 108 dependent
locality/setting views. All-event violations fell from 10 to 3. It did not
transport the conditional harm limit to all six readout localities.

Net easy preservation is a different result: all 18 settings pass and worst
degradation falls from 0.730345% to 0.379671%. Gains on some easy examples can
offset harm on others. Net preservation is not a bound on positive harm.

Across six source-role assignments, averaging three seeds within locality,
grid calibration changes all-ADE relative to the unchanged full neural rule
by -1.378161% to +0.000124%. Five locality-bootstrap intervals are negative;
one overlaps zero; none is positive. The tiny positive point estimate is not
an improvement claim. Individual-seed ranges are in results.md, not substituted
for these seed means.

## Stronger Rescaling Does Not Resolve the Tradeoff
Full neural population rescaling raises complete observed risk passes from
0/18 to 2/18. It eliminates all-event violations in this readout, but leaves
38/108 easy positive-harm violations. Net easy degradation is zero. Three-seed
all-ADE changes are -2.378341% to -0.433005%, with six negative intervals.
The loss in useful interventions is therefore measured, not assumed.

Ridge is still an essential control. Full ridge rescaling reaches 5/18 complete
passes; motion-only ridge reaches 10/18 and eight remaining easy-harm violations.
Neither passes all settings, and selecting its best setting after this readout
would be another development choice, not confirmation. Pure reference fallback
passes observed risk in all settings but makes no additional R-to-P intervention.
It gives up accuracy and is not a learned positive-transfer result.

## Why the Failure Is Credible
Matched-label-support diagnostics separate prediction from missing-label effects.
For the raw full neural rule, median predicted selected positive harm is only
0.300808 times realized harm, while predicted reference mass is 1.494471 times
realized mass. Both biases make the risk ratio optimistic. Even after population
rescaling, median selected-harm coverage is only 0.541182. Four-site moment
matching is not conditional, selection-aware transport calibration.

The within-source leave-one-C diagnostic already exposes instability: neural
full maps pass 59/72 held-C views; ridge full passes 66/72. These overlapping
views do not create additional independent scenes. Finite-scene sensitivity
also rules out claiming a distribution-free 2% certificate from these counts
without substantial additional assumptions.

## Decision
No deployment is promoted. No new neural dynamics, independent risk guarantee
or submission-readiness claim is supported. Keep the 12 reserved calibration
and six confirmation localities closed. Next, diagnose support and selected-harm
transport on source-only held rosters, with ridge retained and the 2% estimand
unchanged; freeze any new support-aware policy before another development
readout. Do not keep shrinking thresholds on these six opened outcomes.

[All rules and CIs](results.md), [transport figure](calibration_transport.svg),
[failure taxonomy](failure_analysis.md), [statistical feasibility](finite_scene_feasibility.md),
[reproduction](operation_zh.md). Image pixels and annotation steps only; no
metric, seconds, physical-safety, human-gold, true3D or foundation claim.
Stage5C and SMC remain off.
