# Frozen-Cap Attribution Data and Model Card

## Research Role
This is a mechanism analysis of expected gain/harm prediction over causal
motion forecasts. It is not a new trajectory predictor or a deployed safety
policy. Its parent source-development results motivated the hypothesis, so
neither preregistration of these contrasts nor fresh arithmetic makes the
source data an independent confirmation set.

The existing twelve European source localities rotate across producer and
controller assignments. Each new view retains its outer exclusion and inner
fitting-locality scoring exclusion across the complete producer chain. There
are six assignments, three cached neural seeds, two forecast families, four
outer contexts and three inner scoring localities: 432 dependent views.
Their windows and assignments must not be counted as independent experiments.

## Data and Inputs
Eight observed and twelve requested future annotation samples, raw frame-ID
stride 12; image detector-box pixels. The interval is query-84 to query+144,
not verified seconds. Future labels are allowed only for supervised targets
and retrospective evaluation. Missing paired targets remain missing.

The frozen probes consume causal scores, forecast disagreement, prior motion
summaries, ordered observed motion and optional past neighbor features. No
future endpoint, future event label, central velocity or held-out endpoint goal
enters inference. All learned transformations and coefficients come from the
parent fitting-only models; no new preprocessing statistics are estimated.

The causal envelope is the maximum pointwise distance between the two
twelve-step forecasts, not their mean distance. It therefore bounds the
matched ADE error difference for any evaluated subset of future steps by
the triangle inequality, without reading that future-validity subset at
inference. This geometric error bound is not a physical-safety guarantee.
Past-only checks concern the exported annotation rows and model access. They
do not establish the annotations' sensor-time availability or an online
perception guarantee.

## Model Intervention
Recover 1,728 unprojected scalar score vectors from fixed ridge models. Compare
four predefined output projections for every input arm. No weights, thresholds,
ridge penalties or model choices are fitted or selected this turn. Signed
scores are an algebra diagnostic; nonnegative uncapped scores can exceed the
geometric envelope. Envelope-only scores can violate easy-harm <= all-harm.
The coupled version raises all-harm and must face its own error guard. It does
not enforce every possible joint moment relation or prove calibration.

All outputs are shadow analyses. No intervention is sent to a trajectory
controller and no new deployment floor is claimed. Cached Torch forecasts
and risk heads remain unchanged; no new neural gradient updates occur.

## Evidence Limits
The main outcome is expected easy-harm MSE, with tail-mass capture,
log-coverage and all-harm MSE guards. It is not t50, trajectory ADE/FDE,
easy-case trajectory degradation or real physical harm. Recovered scores
are frozen before projection scoring, but this is still development analysis.
Cost contrasts use finite paired targets and a positive causal envelope,
exactly as in the parent screen. Unknown targets are not recoded as zero;
these supported-row results do not establish risk for unobserved outcomes.

The paired bootstrap uses four locality-level values per assignment after
averaging outer-context and seed replicas. Six assignments overlap, intervals
are unadjusted, and sparse events limit interpretation. Undefined guard
quantities are not removed. Realized labels can exceed an unbiased expected
cost; a large pointwise ceiling floor does not establish conditional-mean bias.

Independent selection, calibration and confirmation stay closed. No verified
metric scale, seconds-level, human-gold, true3D, foundation, physical-safety
or submission-readiness claim. Stage5C and SMC remain disabled.
