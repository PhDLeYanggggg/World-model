# Cap-Event Auxiliary Supervision for Nested Forecasting Costs

## Material Passport
Source-development code experiment, registered before new fitting/support and
readout. Parent cap-event probes are complete and verified, but their complete
MLP gate failed. Full-input proper-score/ranking signal motivates this repair;
it does not establish cost accuracy, policy utility or independent confirmation.

## Hypothesis and Matched Controls
Does a producer-relative cap-event auxiliary objective improve H and H_E
expected-cost prediction from the same causal inputs, beyond cost-only training
and an equally weighted, locality-shuffled auxiliary task?

Three newly fitted arms: cost_only, cap_aux, shuffled_aux. All use399 causal
inputs from the parent, a shared SiLU32 encoder, two nested cost logits and
one event logit (12,899 parameters). Parameters, initialization, site-balanced
minibatches and2000 updates are matched. The only within-experiment change is
the auxiliary objective: disabled, true event, or event shuffled within each
fitting locality with a fixed seed. The shuffled labels retain each locality's
prevalence and missingness. No hyperparameter, threshold or best-arm search.

Both new cost outputs are direct functions of the encoder:
H = envelope * sigmoid(a); H_E = H * sigmoid(b).
The event probability never multiplies either cost. Reference-cost components
D/D_E remain the frozen original predictions. No global cap relaxation and no
new forecasting or deployment rule. The zero-reference guard is unchanged.

The original frozen cost estimator is an additional strong comparison, not an
identical cost-only arm: it has different inputs and a four-cost objective.
Only the new matched arms isolate auxiliary supervision. Full versus motion
families change the forecast pair and outcome population, not just features.

## Inputs, Targets and Exclusion
Use the existing native causal vector383 + context7 + missingness7 + risk2.
The two risk inputs use existing row-locality-excluded inner producers during
fitting and the frozen outer producer at readout. The already fitted cap-event
classifier is NOT stacked as an input. This avoids a deeper, unverified
cross-fitting chain. No observed future, realized error, event/easy label,
endpoint goal or source ID is an inference feature.

Cost targets use the original three-fitting-locality positive-CV easy cut.
The auxiliary label is common-meta easy harm greater than the existing
row-excluded inner all-harm prediction. It is supervision only. The held
event uses the corresponding frozen three-locality outer estimate. Inner
two-locality versus outer three-locality transport remains a limitation.
The source forecaster excludes all four risk-model localities.

Fitting uses known positive-disagreement rows; exact zero-disagreement cost
predictions are structurally zero and do not establish learning. Positive-CV
easy labels do not cover perfect-CV cases; low event probability cannot relax
their separate zero-reference protection. Unknown labels stay unknown.

Fit imputation/scaling and equal-locality weights only on fitting rows. Keep
the parent input clipping[-20,20]. The existing fitting reference cost scale
is retained; per-output RMS of the normalized H/H_E labels scales squared
loss, floor1e-4. Optimize mean two-output scaled MSE plus auxiliary BCE with
coefficient1 (or zero for cost_only), AdamW lr0.0003/weight_decay0.0001,
batch256/clip5. No class reweighting. Initial cost fractions match fitting
means, event intercept matches fitting prior, final weights start at zero.

## Fixed Matrix and Evidence Rule
Six producer/controller assignments, seeds17/29/43, two forecast families,
four outer localities:144 views,432 heads,864000 fixed updates. Retain all.
Fit support precedes training; all predictions are committed before source
held readout. Model/optimizer/RNG checkpoints support exact continuation.

Primary cost comparisons: cap_aux versus cost_only and versus original.
Report positive-disagreement H_E MSE percentage gain, all-row H MSE gain,
top10% easy-harm mass capture and absolute log coverage-error reduction.
Coverage here is predicted/actual mean cost, not conformal coverage or safety.
Report all four cost-component MSEs, true cap-event probability scores and
fixed-batch cost/auxiliary losses. Event scores alone cannot pass cost gates.

Average the three seeds within locality; use3000 paired resamples of four
localities for each assignment. Assignments and overlapping windows are
dependent. Intervals are exploratory, without multiplicity adjustment, on
previously exposed source development. Missing support is not_estimable,
never silently dropped. No independent-confirmation interpretation.

The primary gate requires all six full-input H_E MSE intervals positive versus
both cost_only and original. Guards require no negative or missing intervals
for tail capture, coverage-error reduction and all-H MSE against both.
A separate task-information gate requires all six full-input H_E MSE intervals
positive against shuffled_aux. Claim an auxiliary contribution only if all
three gates pass. Even then, policy utility, independent confirmation and
deployment remain unproven and unchanged. Do not relax these gates after readout.

## Runtime and Claims
Native arm64 .venv-pytorch,CPU4/interop1/workers0, no resource probing. Pilot
first100 updates of cost_only, then resume that checkpoint to2000. Save and
heartbeat every200 updates. Preserve10GiB disk; choose local/CREATE from
observed pilot cost, never stop merely because training is slow.

Independent selection, reserved calibration and confirmation stay unopened.
Obs8/pred12 native annotation steps, detector-image pixels. Here harm means
forecast error increase, not injury or physical safety. No metric/seconds,
human-gold, true3D, foundation or submission-ready claim. Stage5C/SMC stay off.
