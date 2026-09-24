# Source-Only Nested Forecast Experiment

2026-09-24. Registered before European Squares prediction errors or training
losses are opened. This is method development within the12opened training
localities. It is not reserved model selection, risk calibration or confirmation.

## Question and Fixed Comparison

Does the existing past-context Transformer learn useful dynamics on unsmoothed
detector tracks beyond a strong causal baseline? Does its source-excluded output
provide a valid training basis for subsequent gain/harm intervention learning?
The support intake is complete; there is no reason to keep repeating data audits
instead of performing this falsifiable comparison.

Use the exact registered318,969-target cohort,8observed/12requested raw stride12,
raw image-pixel box centers. Keep unknown/partial labels and original query
sampling. No fitted teacher, external weights, scene images, endpoints, goals,
site ID or future-valid mask enter inputs. Existing SDD outcomes are not mixed
with these new-source scores. The primary SDD protocol does not change.

Compare constant position, last-difference CV, velocity decay0.90/0.97per requested
step, and OLS velocity from the last4/8observed points. OLS estimates only the
past slope and anchors the forecast at the observed current position. This
controls detector jitter without retrospective whole-track smoothing.
The fitting sites select a single baseline by equal-site mean ADE relative to
CV; this baseline initializes and anchors the neural output. All six baselines
are reported. A future-label oracle is diagnostic only, never an input/policy.

Reuse the existing64-wide,two-layer,four-head Transformer and motion-bounded
baseline-relative wrapper. Select up to8nearest complete-history neighbors from
current/past coordinates, tie by scoped agent ID. Short-history neighbors remain
in the source cache but this fixed architecture masks them. Report that limit;
do not claim complete multimodal context. All observed tokens are past-only;
future queries attend only that memory, not future labels or future tokens.

## Nested Source Folds

Create3internal groups of4localities. Sort the12training localities by decreasing
registered past-support count, tie by key; within each successive block of3,
sort SHA256 of `fold_salt|locality_key` once and assign groups0/1/2. Never search
another salt or split after scores. Every recording/season of a locality stays
together. These are internal training cross-fits, not newly independent tests.

For each seed17/29/43, fit six independent fresh predictors:
three singleton-group fits and three two-group-complement fits. Each singleton
fit predicts only the other8localities. Each complement fit predicts only its
excluded4localities. There are18fits,total72,000optimizer updates.

For a later risk head evaluated on groupA, train its cost labels on B/C using
the predictor fitted only onC/B respectively. Its final candidate onA comes from
the B+C predictor. Thus neither upstream predictor nor risk head is trained onA.
The4-versus8-site producer-size difference is an explicit cross-fitting limit,
not ignored. Baseline selection, loss normalizers and all learned preprocessing
use only each predictor's own fit sites. Never use the final held group to choose
the predictor, cost-head hyperparameters or risk thresholds.

## Training and Runtime

Every model gets4,000updates,batch64,AdamWlr0.0003,decay0.0001,cosine schedule,
gradient clip5. Use equal-site sampling, then uniform past-eligible target within
site. Mask unknown supervision; multiply each supported row's loss by indexed/
supported count so this sampler targets the intended supported-site mean ADE.
Divide native ADE by that training site's selected-baseline mean ADE. Keep
scaling constants in fit provenance; no held-target normalization.

Fresh initialization, no early stopping or validation-best selection in this
fixed-budget source experiment. Checkpoint model/optimizer/RNG/draw counts every
200updates and heartbeat/loss every50. First run100updates of the first trial
as a real timing/resume pilot; it belongs to the fixed budget and is not scored.
Then finish all18before new forecast readout. Native arm64 CPU4/inter-op1/workers0,
no Torch resource probing or DataLoader subprocesses. A healthy run is not
stopped for slowness. Estimate local feasibility from that pilot; CREATE would
require a located M3W directory and allowed scheduler path, not the simulation
project. Do not silently reduce steps or substitute NumPy training.

## Fixed Readout and Limits

Primary: equal-locality native masked ADE relative to each complement-selected
strong baseline and to CV. Also report all12-step FDE where its endpoint exists,
complete-future sensitivity, training-complement positive-CVq25 easy,
training-complement CVq75hard, exact-zero-CV absolute harms, unknown coverage,
per-locality errors/tails and every seed. Do not pool camera pixel errors into a
common physical-scale claim. Average seed errors, not forecasts; use3,000paired
locality bootstrap resamples and disclose shared fitting/exploratory limitations.

Save predictions and producer identities before computing metrics. Reproduce
the readout from frozen prediction bytes; verify a fixed raw checkpoint inference
sample and exact interrupted/resumed training behavior separately. A hash alone
is not evidence the model learned anything. Negative scores remain visible.

This experiment does not tune or deploy a risk policy. A follow-on source-only
gain/harm study may use the nested predictions under its own fixed registration.
Do not open the6selection,12calibration or6confirmation groups to repair a
negative result. The2%positive-easy limit and zero-reference harm allowance0
remain. No formal safety/independence/physical-time/metric/3D/foundation claim,
Stage5C, SMC or manuscript submission is authorized by completion of these fits.
