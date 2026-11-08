# CV-Anchored Forecast Selection With Out-of-Location Abstention

This is a follow-up development diagnostic on the already read OpenDD v3
cohort. It is not new independent confirmation. Freeze this comparison
before scoring. Use the identical 8/12 stride-12 past-only query schedule,
pedestrian targets, missing labels, seven location folds, and three seeds
from the OpenDD location diagnostic. No change to DroneCrowd or its 2% rule.

Use last-displacement CV as fallback. Candidates are OLS4 and the original
four motion alternatives. Train matched CV-only and dual-reference models:
same features, draws, parameter count, optimization and CV supervision.
CV-only duplicates CV supervision in the auxiliary head, while dual uses
OLS4 supervision there. Each head predicts paired signed ADE gain and
positive harm, averaged across available steps and normalized by the same
past motion scale. The CV score is predicted CV gain minus CV harm. The
dual score also deducts predicted OLS4 harm. OLS4 self-harm is exactly zero.
Neither future availability nor positions are score inputs.

For each outer held site and seed, fit six inner predictions: hold each
other site out of a fit on five sites. Reuse the same five-site fit for both
directions of each excluded pair. Select the smallest threshold among ties
from [0, .002, .005, .01, .02, .04, .08] by mean paired lower-bound gain
against CV, provided gain is positive, there is at least one edit at three
or more inner sites, and signed easy ADE change is <=2% at every inner site
with defined support at all six. Otherwise return exact CV everywhere.
The descriptive easy cut is the positive CV ADE 25th percentile of complete
rows in each fit's training sites. No threshold or feature is fitted on the
outer site. This diagnostic filter is not the original per-view safety gate.

Train each outer model on the other six sites. Report at most one edit per
query using the frozen inner threshold; also show forced-one selection,
minimum change, history hindcast and an observed-label oracle, all at exact
matched forced counts. The calibrated policy is compared to no edit and to
both simple rules on the same edited queries. No fixed-total-count rule is
used to select the abstention policy. Arms use the same forecasts and their
gain differences are paired stepwise with missing disagreement debited.
Report observed ADE/FDE, lower gain, signed easy change, positive easy
harm, switches, per-site results and 3,000 equal-site bootstrap resamples.
Run native arm64, four compute threads, zero workers, 300 updates per fit
with checkpoints every 100, and no outer-site selection or deployment.
