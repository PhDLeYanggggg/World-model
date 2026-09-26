# Fitting-Defined Causal Context and Cost-Residual Probe

## Material Passport

Parent8cdc8de0 is verified locally/on GitHub:48 artifacts and9 source bindings.
The previous turn made progress but neither the joint recording-influence
nor radial-support flag passed. This next source-development experiment asks
whether simple conditional residual structure transfers across localities.
It does not open independent selection, risk-calibration or confirmation.

## Fixed Features and Models

Use seven allowlisted summaries: past net displacement divided by max(past
path length,current box width); signed speed change divided by max(mean past
speed,width/7); mean absolute past turn angle; current visible neighbor count
(capped at8 by the inherited input); nearest current distance/past scale;
that neighbor's latest closing displacement/past velocity scale; and frozen
rollout disagreement/past scale. Neighbors use existing past masks. Missing
neighbor distance/closing values form explicit missing bins. No future input,
central velocity, future track length, endpoint goal, identity or held-statistic
feature. The historical name path_efficiency denotes the width-stabilized
net-displacement ratio, not literal net/path when width exceeds path length.

For each of144 full/motion-only source-held views retain all three frozen
estimators: original matched cost-only, ordinary auxiliary and severity
auxiliary. Exclude the held locality across the entire fitting chain.
Set tercile cuts from known positive-disagreement fitting rows only, with
the inherited equal-locality weights renormalized on that support. Duplicated
cuts and explicit missing bins stay in the schema. No cut/hyperparameter search.

Fit exactly two closed-form residual probes per frozen estimator: global
intercept and additive context-bin ridge. The latter has an intercept plus
four indicators for each of seven features, ridge0.1 on non-intercept terms.
Normalize the response (true easy-harm cost minus frozen predicted cost) by
fitting RMS easy-harm target, floor1e-8. Predict H_E as frozen H_E plus learned
offset, clipped to[0,frozen H]; leave D,H,D_E unchanged. This is a cost-head
probe, not trajectory correction, policy selection or neural training.
864 closed-form fits, zero new Torch optimizer updates. Frozen Torch models
perform inference in native arm64 CPU4/interop1/workers0.

Base predictions used to fit residuals are in-sample for that base estimator;
this is a known limitation, not independent calibration or OOF stacking.
Never reuse another held-fold model that trained on the outer held locality
to manufacture OOF residuals. Before interpreting a positive result for a
pipeline, a separate nested-OOF experiment is necessary.

## Freeze and Readout

Commit registration before pilot/fitting. Save all coefficients, exact input
hashes and held prediction hashes; commit prediction freeze before new held
readout. All source-development labels were already exposed historically;
the freeze does not make them independent. Pilot is reused, not substituted
for the full36 groups. Immutable receipts, heartbeat, resume by view and a
10GiB free-space reserve; no remote jobs or checkpoints changed.

Retain raw and corrected cost metrics, the global-only control and the
original strong estimator. Reuse the parent all-known/positive-disagreement
metrics, top10 harm-capture and coverage guards. Average the three seed
contrasts within each held locality, then3000 bootstrap draws of four
localities per assignment. Keep all six assignments and both input pairs;
no favorable-role selection or window-independent CI. These are dependent,
conditional-development intervals without multiplicity correction.

Primary diagnostic criterion: full/original context probe improves easy-harm
MSE against both the original and the global-intercept control in all six
assignment intervals, with no negative/missing original-comparison top10 or
coverage interval. Other estimators are robustness comparisons, not substitutes
for this fixed primary. A diagnostic signal cannot authorize deployment.

For each feature/bin report centered residual direction on known positive-
disagreement rows for each of three fitting localities and the held locality.
A descriptive supported cell needs
at least20 rows and10 recording-agent tracks in every locality. Consistency
requires the same nonzero sign across all three fitting sites; then count
held same/opposite sign. Values within1e-8 of zero are numeric ties. These
counts use held labels only for analysis, never prediction or threshold fit;
overlapping bins/views are not independent tests or causal explanations.

## Boundaries

No forecast/policy change, reserved-role access, test tuning, metric/seconds,
human-gold, physical-safety, true3D, foundation or submission-ready claim.
Eight observed/twelve predicted annotation steps, detector-derived pixels.
Stage5C and SMC off. Raw data, caches, per-row predictions and weights private.
Full legacy suite may reuse verified same-version results; fresh scoped tests
and every probe/checkpoint/prediction replay must be labeled accurately.
