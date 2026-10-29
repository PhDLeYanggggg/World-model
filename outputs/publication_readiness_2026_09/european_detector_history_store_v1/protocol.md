# Original Detection Histories: Input Alignment Before New Training

## Material Passport

Planned fresh_run: reparse original approved raw CSV members, recover observed
confidence and exactly align eight past boxes to existing model rows. Existing
source roles, raw archive and input caches are cached_verified. New training,
forecast/policy scoring and independent evaluation are not_run here.

## Scientific Scope

Previous coupled-cost and stratified controls fail complete selected risk.
The current SOURCE geometry mostly uses box-midpoint motion, not temporal
box shape/confidence. The older source_box_motion experiment uses SDD image
flow, not these European raw detector histories; retain its negative result.
This extraction tests availability/alignment, not whether the new information
improves forecasting or risk. No homography is required. Existing midpoint
targets, obs8/pred12/stride12, source splits and 2% risk rule do not change.

## Allowed Rows And Inputs

Use exactly the74 unique fitting recordings and149,510 unique packed rows
identified by the verified static-context audit. Require their source_training
role before opening a CSV. The raw archive hash, full structured-row hash,
source input receipt hashes and all accessed arrays are verified. No reserved
CSV, future target array, future support mask or independent/DroneCrowd role
is opened. Union feature extraction across contexts is not a global learned
normalizer; each later model must fit preprocessing on its own fitting rows.

Match packed recording/frame/agent keys to original current-visible input
keys, then recover exactly query_frame minus[84,72,60,48,36,24,12,0]. Every
lookup must exist and its four raw box coordinates must exactly equal the
stored input_history_boxes. The midpoint and last width must exactly match
the existing packed history/width. Never fill missing observations, remove
queries using future labels or extend the observation length.

The64 fields comprise eight past values for each of: log1p box width, log1p
box height, detector confidence, regularized log aspect, half-height step,
body-scale center dx, body-scale center dy, consecutive box IoU. Steps have
an explicit zero at the first observation. Body scale is one pixel plus the
median observed height, so zero-area boxes remain finite and are not dropped.
These are pixel/body-relative features, not metres, seconds or intent labels.

## Causality And Controls

Only current/past supplied detections contribute. Unit tests poison/truncate
future raw rows; a separate per-record real check repeats first/middle/last
query lookup using a truncated raw prefix. No future label defines input or
query inclusion. This is causality relative to supplied exports, not verified
live sensor-as-of tracking or human gold.

The current-only representation repeats the last observed box/confidence,
while deterministic64-dimensional noise is computed only from scoped current
recording/agent/frame keys. These are future candidate controls, not trained
results. Their normalization and model selection must remain fitting-only.

## Runtime And Verification

Register this code/configuration before a one-record pilot. Time raw loading,
alignment, memory and cache size; proceed through the complete roster rather
than downsampling for speed. Stream the ZIP member without extracting raw
files. Save per-record array checkpoints atomically and use --resume, a
process lock, heartbeat,10GiB reserve,256MiB private-cache cap and12-hour
boundary. Current1.67GB uncompressed fitting members are already local.
No CREATE job is warranted unless actual local costs require it.

After complete extraction, a separate --verify reparses every selected raw
member and checks all confidence/features/keys and reports exactly. Only
aggregate receipts/schema are public; private arrays and raw data stay out of
Git. Prefix checks are implementation evidence, not scientific model success.
No deployment, metric/effective-second, true3D, foundation, Stage5C or SMC claim.

## Execution Repair Before Raw Access

The first pilot invocation fails before raw-archive access: a local receipt
variable named runtime shadows the runtime function. A regression reproduces
that exception. Registration_v2 renames the receipt and adds the startup
regression; it retains the original registration/failure record. No raw row,
feature, target, role, query roster or numerical extraction rule changes.
