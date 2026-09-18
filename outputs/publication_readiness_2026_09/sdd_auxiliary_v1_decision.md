# SDD Auxiliary Training: Approved Source and Fixed Comparison

## Scientific Decision

On 2026-09-18 the project owner replied "按这个方案继续" to the explicit
recommendation: original SDD train-40 only, stride 12 raw frames, eight observed
points and twelve predicted points, as a separate auxiliary arm with a matched
no-SDD control. ETH/UCY primary, roles and split remain unchanged. This resolves
the auxiliary source-role/sampling blocker, not independent confirmation.
The last future offset is +144 raw frames. No seconds or metric equivalence is
assumed. Source histories retain offline annotation/interpolation provenance.

## Falsifiable Comparison

Does supervised SDD pretraining improve held-fit-scene forecasting with preserved
easy cases, and does any transfer require actual image content rather than image
support masks? Fixed 2 x 3 design: source schedule (no_aux, sdd_aux) crossed with
geometry, mask_only and past_rgb. Geometry keeps mean past-image support as in
the existing model; mask_only adds a learned spatial support representation;
past_rgb adds retained observed pixels. All use OfflineVisualForecast, identical
initialization by seed, with no new architecture, teacher or learned goals.

Three seeds 17/29/43, three existing physical fit folds ETH/Hotel/grouped Zara.
All 11,966 main fit windows retained. Training never opens the held fit fold's
labels or estimates preprocessing from it. Repeatedly explored fit sites remain
exploratory; Students/development/calibration/confirmation stay closed. No final
model or threshold is selected from this experiment.

Each fit has 2,000 pretraining updates followed by 4,000 main-training updates.
The auxiliary schedule pretrains on SDD; the matched no_aux schedule pretrains on
its main training fold. Both then reset AdamW and the main minibatch RNG and
train on the identical main minibatch stream. Model weights are retained across
phases. Batch64, learning rate0.0003, weight decay0.0001, gradient norm cap5,
final checkpoint only, no early stopping or held-score tuning. All54fits are new;
324,000 total registered updates. Three training seeds do not add independent sites.

Source pretraining uses all229,333past-eligible stride12queries in the original
40trainvideos. No cap, event-based row selection or future-completeness filter.
Targets/masks are separate from model inputs. Log1p per-row masked ADE is averaged
over supported rows in the batch; no-label rows remain indexed, do not enter the
loss denominator and are not zero-error evidence. An all-unsupported batch gives
zero loss and is recorded. Main training uses complete labels and the same
log1p-ADE surrogate; primary remains mean past-normalized ADE, equal physical
scene aggregation. This does not redefine the primary with masked auxiliary labels.

The existing476-column geometry schema and native raw-frame feature conventions
are retained in both schedules. Normalization moments come only from each main
training fold, are shared across schedules/modalities, and zero unsupported
constant columns after standardization. No SDD-dependent preprocessing change is
confounded with the source contrast. This is not a claim of full coordinate/time
invariance; cross-source covariate shift and clipped features must be reported.

RGB uses the existing96-to32 crop/pooling convention. SDD image boxes use the
verified reference-to-video scale mapping and current-frame inferred black-border
mask, never a test-endpoint goal. All requested past crops are built, not only
the old5,074diagnosticjoins. Main images reuse their unchanged supplied-H/native
index mapping. Zara03 lacks video and retains its explicit zero image support.
Different views/resolutions/interpolation remain domain limitations.

## Evidence and Failure Rules

Report every source/modality/seed/fold: primary ADE/FDE vs causal CV and the
training-selected strongest baseline, native-coordinate diagnostics by recording,
easy relative and absolute harm, stationary/movement event slices, tail error,
training losses, sampler counts, runtime and train/held gap. Primary auxiliary
contrast holds modality fixed. Pixel contribution compares past_rgb to mask_only,
not only geometry. Include paired2,000physical-scene resamples over only the three
exposed fit sites, explicitly descriptive and conditional on fixed fits.
No oracle, positive direction or source engineering result can promote deployment.

Save atomic checkpoint including optimizer/sampler/Torch RNG, identity and input
hashes every200updates and phase boundary. An interruption resumes the last
atomic checkpoint, not a possibly half-applied optimizer step. Heartbeats include PID,
phase, step and loss. A100update pilot resumes the same registered fit without
held evaluation. Nativearm64CPU4/interop1/workers0 first; estimate cost from
actual updates. Do not stop for slowness or silently reduce budget. No Stage5C,
SMC, metric/seconds, foundation, true3D or independent-test claim.

Freeze code/config before real fitting. Verify source-role rejection, future-input
invariance, masked gradients and exact phase-boundary resume with focused tests.
After completion replay saved predictions and verify completed resume adds no
updates. Source data, image cache and checkpoints remain private; Git receives
only code/config, aggregate reports and author-voice project status.
