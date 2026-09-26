# Causal Context Cost Probe: Completed, Primary Gate Failed

## What Was Run

All 864 fixed closed-form probes completed on 432 frozen estimator instances
across 144 aligned source-held views and 36 groups. There were zero new Torch
optimizer updates. Registration b7a829eb preceded fitting; prediction freeze
3ac1ae56 was pushed before the new source-held readout. The six-probe pilot
took 13.9611 group seconds; the full fit reused it and recorded 313.8133 run
seconds excluding parent preflight. These are not end-to-end experiment times.

Fresh work comprises the seven-feature context extraction, two probe fits,
new held predictions and paired readout. The underlying producers, neural
estimators and source data are cached_verified, not freshly trained. All
independent selection, risk-calibration and confirmation roles remain unopened.

## Fixed Primary Result

The outcome is easy-harm cost-estimation MSE on positive-disagreement rows,
not FDE/ADE, trajectory gain, intervention success or physical safety.
Here H = max(candidate ADE - reference ADE, 0), and H_E = H times the
indicator that positive CV ADE is at or below the fitting-only 25th-percentile
easy cut. These future errors are supervision/readout labels, never features.
The probe changes predicted H_E only; it does not change either trajectory.
Positive/negative below means a 95% locality-bootstrap interval wholly above/
below zero. Three seeds are averaged within locality before 3,000 resamples
of four localities per assignment. All six dependent assignments are retained.

| Comparison | Positive / negative / overlapping intervals | Point range, MSE improvement |
|---|---|---|
| Original + context vs original, full | 2 / 0 / 4 | -1.8796% to +0.9351% |
| Original + context vs original + global bias, full | 2 / 0 / 4 | -1.1155% to +0.6679% |
| Original + global bias vs original, full | 1 / 0 / 5 | -1.1112% to +0.2699% |
| Original + context vs original, motion-only | 0 / 0 / 6 | -16.5327% to -0.5338% |
| Original + context vs original + global bias, motion-only | 1 / 0 / 5 | -8.2746% to +5.2874% |

The predeclared primary criterion requires six positive intervals against
both original and global-bias controls. It fails. Full-input top10 harm
capture has one positive and five overlapping intervals against each
control; coverage-log-error reduction has three positive and three overlaps.
The registered no-negative/no-missing guard passes, but this is not proof of
noninferiority or safe deployment. Some tail-capture point estimates worsen.

Ordinary-auxiliary context correction has zero positive intervals against
the original in full inputs; severity-auxiliary context has one, with point
range -21.0521% to +1.1920%. The motion-only severity comparison retains a
negative interval and a -51.3643% worst assignment point. Neither auxiliary
arm replaces the primary comparator. All eleven comparisons and both input
pairs are retained in aggregate_metrics.json and results.md.

## Conditional Pattern, Not a Mechanism Claim

For the full-input original estimator, 120 supported closing-speed cells
have matching nonzero residual signs in all three fitting localities;
102 retain that sign in the held locality and 18 reverse. Speed change gives
96 same /29 opposite among 125 fitting-consistent cells. These are dependent,
overlapping cells from repeated views, not 120 or 125 independent replications.
The residual is centered within each locality, so this is relative local bias,
not agreement of absolute harm levels or calibrated costs.

Other features are less consistent: turn angle gives 17 same /23 opposite;
rollout disagreement gives 31 /30. Motion-only closing speed gives 29 /25.
No feature was selected or removed from the completed probe using these
held outcomes. AUROC improves in four full-input original-context intervals,
but event ranking does not establish accurate harm magnitude.

## Decision

Keep the original frozen strong cost estimator and all deployment policies
unchanged. Do not advance independent calibration or confirmation. The next
controlled question is whether the in-sample residual construction conceals
transportable error structure: register a nested, fitting-locality-OOF probe
with the same seven summaries and fixed global/context controls, subject to
an explicit audit of inner-fold target, threshold and preprocessing lineage.
Do not reuse models exposed to the outer held locality or tune new bins from
these results. The next experiment is not_run in this record.

This remains exploratory source-development evidence, not independent
confirmation, a repaired world model or a CCF-A submission candidate.
Eight observed/twelve predicted annotation steps; detector-derived pixels.
No metric/seconds, human-gold, physical-safety, true3D or foundation claim.
Stage5C and SMC remain off. Private data, predictions and weights stay local.
