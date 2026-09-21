# Native Cost-Learning Producer Repair

Registered before new fitting, 2026-09-21. This is a source-only, exploratory
repair of upstream fitting exposure, not new independent calibration/testing.
The author has resolved the metric choice. The zero-CV native-risk tolerance
has been asked separately; this prerequisite does not choose or relax it.

## Why This Experiment

The preceding matched native-loss predictor improves mean source ADE by 7.63%
over CV but harms some easy queries. A risk head needs honest gain/harm labels.
Simply splitting the existing single-held-site OOF cache is not enough: a
training row's producer may have fitted the proposed outer validation scene.
We must exclude both that outer scene and the row's inner scene upstream.

Keep the admitted four SDD training sites, source row index, observation/task,
native metric and prior closed-role boundaries unchanged. No original validation,
test, main, external or bookstore readout. These sites have already influenced
research design; a clean fitting lineage does not make them untouched test data.
Native loss is chosen for follow-up after the disclosed source comparison, not
as an outcome-independent choice. Record this design exposure separately.

## Fixed Fits and Reuse

Use all six unordered pairs of excluded physical sites, seeds 17/29/43:
18 fresh random-initialized fits, each 4,000 updates, total 72,000. Every fit
trains on exactly the remaining two sites. No warm start or fitted parent.
Reuse the existing 88,514-parameter Transformer, input conditioner, motion bound,
native loss, site-uniform sampler, optimizer and schedule without modification.
Loss normalizers and missing-support corrections are recomputed on those two
training sites only. Pair order must not change a fit or its batches.

First run 100 updates of coupa/deathCircle/seed17 to verify the new exclusions
and resume path in a real training process. That pilot counts toward its fixed
4,000-update budget. Local native arm64 CPU4/interop1/workers0 is justified by
the preceding 33.40-minute real 24-fit matrix; check for existing jobs and disk
space before launch. Expected new fitting cost is about 25-40 minutes plus cache
construction. Do not edit bound files or restart a live process. Atomic
checkpoints and 50-update heartbeat preserve recovery; resume checks identities.

Reuse, without refitting or rescoring, the twelve existing native-loss outer-held
models and their bound predictions. All three seeds stay separate. The new
pair-excluded models cover 36 ordered outer/inner producer requirements; do not
count reuse of each pair in both directions as two separate training runs.

## Targets and Data Boundaries

After all 18 endpoints are complete, infer on both excluded source sites with
the fixed final checkpoint. Prediction payload is the existing past-only history,
neighbor history, causal CV rollout and requested prediction times. Future
coordinates, future support masks, scene IDs and cost labels are not inputs.

Store prediction arrays separately from supervised cost arrays. The latter hold
native ADE/FDE, gain = CV ADE - neural ADE, benefit = max(gain,0), and harm =
max(-gain,0). Unknown costs stay NaN. All past-eligible row IDs remain indexed;
valid-count/complete flags are target metadata, never inference features.
These are raw costs, not calibrated probabilities or thresholded safety labels.

For each outer scene and seed, the cost-head training view contains only the
other three scenes. Its inner row producer excludes that row scene AND the outer
scene. The existing single-held outer model remains suitable for later prediction
on the outer scene under its declared fitting exclusions. Bind all row identities,
arrays, producer checkpoints, preprocessing sites, source code and data hashes.

The source-specific lineage guard admits only these unparented random-init
producers, explicitly checking fitted preprocessing and no checkpoint-selection
or calibration exposure. It is not a replacement for the general recursive
contract on arbitrary models, and does not certify truthful unknown history.
Record known design exposure to all four sites and independent_confirmation=false.

## Verification and Claims

Test both excluded scenes' target/scale poisoning, source-group identities,
training/normalizer exposure, invalid warm starts, unknown labels and exact resume.
Verify all final checkpoints and draw counts, all cost identities and row
alignment. Replay first/middle/last fixed inference batches from every producer.
Use a separate arithmetic reduction of native ADE/FDE to check saved labels.
Reject altered bytes before labeling an archive usable. Reproduction is not a
new training result and overlapping windows are not independent samples.

This turn supplies 12 clean exploratory head-training views, not trained risk
heads, intervention scores, new forecasting improvement, a safety certificate or
an independent evaluation. Do not transplant old normalized easy thresholds.
Risk-head fitting, native easy/zero-reference rules, model selection, calibration
and eventual confirmation require their own fixed follow-up registration.

Only source/config/tests/reports/light metrics go to GitHub. Checkpoints, row
predictions, labels and caches stay under the ignored data/stage* tree.
Pixel/raw-frame only, not metric/seconds/true-3D/foundation. Stage5C and SMC stay off.
