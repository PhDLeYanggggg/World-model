# Direct Pair Forecast of Exchange Utility

The support-preserving action menu has an outcome-aware safe envelope of
+0.082 points vs fixed half, but the marginal interpolation head captures
almost none of it. Test whether directly learning the exact realized
two-target action closes that gap. Keep the same OpenDD development sites,
seeds, query roster, prior forecasts, one-transfer menu and radius
conservation. Do not change 2% recording risk or read any external data.

Train 21 multi-output tree heads (24 trees, maximum depth 7, minimum leaf
24, 65% feature subsampling) on pair-excluded donor predictions. Each
ordered action receives features of both recipients and their difference,
relative distance, and actual radius fractions. Use labels computed at
the actual candidate radius: paired lower gain, three original risk
numerator changes, and a beneficial/nonincreasing-risk event. Weight
locations, their queries, and the actions within each query equally.

Compare no extra exchange, original marginal risk guard, direct
utility, direct risk guard (primary), and direct risk guard with
predicted event probability at least .75. Compare to the exact
outcome-aware envelope as a diagnostic only. Do not select thresholds,
fit a new forecast, or change any recipient or total movement. Report
recording pass transitions and paired location-bootstrap uncertainty
against fixed half and the marginal guard. Since these are complete
policies, no-edit differences remain valid as zero against fixed half;
conditional matched utility is not imputed for empty views.

A development gain requires a positive lower confidence endpoint against
fixed half and at least as many original risk passes. This is not the
full original rule or external qualification. No confirmation access,
deployment, Stage5C, or SMC.
