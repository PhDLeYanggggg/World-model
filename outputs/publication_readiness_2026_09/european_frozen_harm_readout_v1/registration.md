# Frozen Harm-Magnitude Readout Experiment

## Material Passport

Registered before new readout training and outcome evaluation. Parent mean
and fractional heads, source forecasts and splits are cached_verified.
The preceding membership diagnosis is fresh_run post-hoc source-development
analysis; it is not confirmation. It motivates this test but changes neither
the primary scientific endpoint nor the 2% intervention-risk tolerance.

## Hypothesis

The fractional objective improved harm-event ranking but not harm magnitude.
Outside-easy rows dominate 35/39 full-pair worsening views. Test whether a
separate cost-focused readout can use frozen fractional features better than
an identically fitted readout on frozen mean features. A matched frozen-mean
control distinguishes the feature contribution from extra readout training.

This uses the general diagnostic idea of an independently trained probe on a
frozen representation, described for classification by
[Alain and Bengio (2016)](https://arxiv.org/abs/1610.01644). Reading scope is
the original abstract. Our readout is bounded cost regression, not their
classification experiment; the citation establishes neither novelty nor a
guarantee of cost prediction, safety or transfer. A probe is not a new world
dynamics architecture or evidence of JEPA/Transformer contribution.

## Exactly What Changes

Two arms: mean_features and fractional_features. Extract the 64-dimensional
GELU hidden layer from each already trained locality-excluded risk head,
under no_grad. Keep the encoder unchanged. Normalize hidden features with
equally weighted known rows from the three fitting localities only; standard
deviation floor1e-6, no clipping or test statistics.

Fit a linear64-to2 readout with130 parameters. Its output is
H=e*sigmoid(a), H_easy=H*sigmoid(b), preserving0<=H_easy<=H<=e. Use the
original mean head's delivered D_all and D_easy bit-for-bit in both arms.
The readout minimizes the mean of two squared harm errors normalized by their
training RMS, floor1e-4. No fractional auxiliary, event BCE, selected-group
loss, oversampling, target clipping, threshold fitting or coefficient sweep.
The nested fraction is not interpreted as an easy-event probability.

Initialize both readout weight matrices to zero and biases using the same
training moments/envelope. Initial delivered harm and diagnostic losses must
match exactly. AdamW lr0.0003/wd0.0001, batch256, clip5, 2,000updates. The
same locality-balanced sampler seed17/29/43 and known-label support are used;
draw counts and terminal RNG must match the cached2,000-update mean control.
Extra frozen-feature training is not equal total compute to the original head.

This differs from prior population rescaling/selected-risk grids, warm shared
continuation and frozen-reference protection: the representation does not
update and two identically trained low-capacity readouts compare frozen feature
sources. The causal trajectory pair and original reference moments stay fixed.

## Splits and Budget

Six ordered source assignments x three seeds x two forecast pairs x four
leave-one-B-locality-out folds x two readouts =288 fresh heads,576,000updates.
Three B localities determine feature normalization, cost scale, easy cut,
loss scales and training score bins. Source-A producers exclude B. The fourth
B locality is excluded from every fitting step. Future error is supervision
and evaluation only, never an inference input. No central velocity or test goals.

All historical development exposure remains disclosed. Source C is not used
for policy evaluation or model selection this round. Six opened selection
localities are unused;12 reserved calibration and6 confirmation remain closed.
The inherited context may load cached target arrays, but no held outcomes enter
fitting or selection. This is not a claim of previously unseen confirmation.

## Frozen Readout and Gate

Push all new prediction hashes before current readout. Retain all arms and
original_mean/original_fractional controls. Primary contrast: conditional
easy-harm MSE fractional_features versus mean_features. Also require improvement
over original_mean before any later policy experiment. For each contrast,
compute per-locality relative MSE gains, average three seeds within locality,
then3,000 bootstrap resamples of four localities, seed47131, for each of six
source assignments. No favorable role/seed selection; no multiplicity-adjusted
or independent-confirmation claim.

Advancement requires all six full-pair primary intervals positive AND all six
full-pair original_mean comparison intervals positive. Conditional top10 harm
capture and absolute-log-coverage-error contrasts must have no negative or
unestimable interval for either comparator. Reference moments must remain
exactly unchanged. Motion-only is reported in full and cannot replace a failed
full gate. Even a pass only permits considering a separately registered policy
experiment; it is not deployment permission. No ADE/policy benefit is claimed.

Report fitting losses, all four moment errors, harm ranking, coverage, weak
support and easy-event error partitions. No later stopping or hyperparameter
selection based on these outcomes. Failed comparisons remain public.

## Runtime and Limits

Native arm64 .venv-pytorch, CPU4/interop1/workers0, one process. Real100-update
pilot is included in the first model's2,000update budget and resumes exactly.
Checkpoint optimizer/RNG/normalization and heartbeat/PID. Keep10GiB free disk.
Current local resources fit the earlier controlled run; no CREATE job is needed.
M3W remote directory remains unverified under the existing simulation-project
handoff restrictions. No broad remote scan or changes to other jobs.

Obs8/pred12 annotation steps at raw stride12, detector-derived image pixels.
No metric/seconds, human-gold, physical safety, true3D, foundation or submission
readiness claim. No new forecaster. Stage5C and SMC remain disabled.
