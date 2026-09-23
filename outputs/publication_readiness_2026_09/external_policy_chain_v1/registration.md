# Frozen Predictor / Cost / Joint-Control Chain

2026-09-23. Register before the new complete-chain input probes and before
reserved-source prediction. No new model fitting or outcome readout in this step.

Bind the six completed all-source predictors and twelve OOF-trained cost heads,
their dependencies, feature builders, past-only input adapter and numerical
control solver. Keep both forecast families, both cost estimators and all three
seeds. Do not choose a winner from fitting loss or confirmation results.

Inherit the native-cost strict policy: exact past-stop protection, strictly
positive estimated net gain, and predicted harm at most0.1 times benefit. This
is an empirical rule, not calibrated probability or a statistical certificate.
Reuse native joint-control full and half-count comparisons. The reference sorts
eligible agents by predicted net gain, then integer agent ID. The half count is
floor(0.5*eligible_count); the same reference supplies the harm budget. Compare
independent ranking, unary geometry and joint geometry at the same count and
predicted budget. Whole-scene uniform choice is a separate control and is not
advertised as count matched. At the full count there is no choice among eligible
agents, so this arm cannot establish coupling. The fixed solver limit is5seconds;
nonoptimal or uncertified solutions fall back and are explicitly unmatched.

Use the existing common-frame proximity proxy: radius=median past target scale,
threshold=0.1*radius, pair_weight=1. Costs are in dataset-local coordinates and
divided by the frozen source-head cost scale. This fixed tradeoff can be sensitive
to cross-domain coordinate scale. No external statistics or physical-unit
calibration will be silently substituted. Unknown-context edges remain unpriced
and reported; this is not physical collision avoidance or all-agent dynamics.

All currently visible agents remain in scene context. Only agents with complete
compatible past support become neural targets; others retain an explicit causal
CV validity mask, not invented stationary certainty. Future-label support is
absent from the inference API. The primary mechanism contrast is joint versus
unary geometry at half count; no-correction CV and uncontrolled neural forecasts
remain controls. Preserve failed solver decisions and zero eligible pools.

Run every model/head/seed on first, middle and last source query times from each
of33 source recordings, selected without labels. These99queries test end-to-end
inference, not source generalization. Hash forecasts, scores and decisions and
replay them in a separate process. Also exercise the external stride1 prefix
adapter with synthetic prefixes and altered/absent future tails. No reserved DUT
or DroneCrowd predictive arrays/labels are opened.

The frozen chain is an uncalibrated research instrument, not a deployment or
predictive admission grant. Whole-source role reservations remain binding.
Scientific intake, source-specific observation restrictions, small calibration
site count, exchangeability and independent confirmation remain separate gates.
8/12 native annotation steps do not establish matched physical duration between
SDD stride12 and external stride1. No metric/seconds/true3D/foundation claim.
Stage5C and SMC remain disabled.
