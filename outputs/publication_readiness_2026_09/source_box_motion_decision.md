# Observed Box Motion: Fixed Source Comparison

Registered before fitting; exploratory source development, not confirmation.

## Hypothesis

Past image motion relative to the annotated box's surroundings might retain
information lost by frozen global appearance pooling. This is not a new optical
flow method. The earlier ETH/Hotel/Zara observed/native-resolution flow experiments
were negative. This comparison asks a narrower question on the already explored
SDD stationary-history cohort, using annotation-box rather than fixed-center
regions. Neither annotation regions nor background proxies are human segmentation.

## Input And Measurement

All 15,430 admitted windows, four source sites and 29 recordings are retained.
Eight observed annotation steps, stride12 raw frames, predict12 steps. Main,
bookstore and external readouts remain closed. Source pixels and annotation
metadata are hash verified. No future target is an inference feature.

Fixed OpenCV4.13 Farneback forward/backward flow on the existing32x32 crops;
restore rounded crop-origin translation and video-to-annotation resize per axis.
Require full pixel coverage, forward/back consistency <=1.5 low-resolution
pixels and >=16 consistent pixels per region. Surround excludes a3-video-pixel
box rim. Unsupported measurements remain zero with explicit support flags;
no row is dropped. Box/surround medians, contrast and magnitude quantiles are
proxies, not verified camera compensation or physical velocity.

[OpenCV optical flow documentation](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html).
Exposure, occlusion, low-pass resolution and neighboring people can confound flow.
All labels/boxes are supplied offline annotations, not sensor-as-of evidence.

## Fixed Training Matrix

Two arms: quality-only and quality+motion. Both retain the same480 trajectory
features, observed image coverage and nine quality/support/annotation flags.
Only the latter receives ten measured motion channels per adjacent history pair.
The first of eight tokens is zero, then seven past-pair tokens. Tokens occupy19
of512 channels; the remaining channels are zero. Neither arm uses ResNet features.

Use the same63,960-parameter temporal readout, train-scale pre-bound output
conditioning, original bounded trajectory function, optimizer and exact
importance-corrected uniform-row ADE objective as the preceding experiment.
Train-only channel mean/std, constant-column removal and fixed8-sigma clipping.
Normalize motion with past rotation and existing past native/restoration scales.

Four excluded-source-site folds x seeds17/29/43 x two arms =24 fresh fits;
10,000 updates each, batch64, CPU4/inter-op1/workers0. Same saved episode draw
streams as controls, checkpoint every200, fixed cosine schedule. A100-update
training-only resume pilot is included, not extra training. No early stopping,
seed/model/threshold selection from held labels. Old conditioned geometry and
centered heads are cached_verified references, not fresh fits.

## Readout

Report all endpoints, train/held ADE/FDE, equal-site and window-weighted gain,
nonzero-target gain, static absolute annotation-pixel harm, hard slices and
binary candidate/CV oracle upper bound. Static percentage harm is undefined
against zero-error CV. Paired2,000 bootstrap resamples of four explored sites,
averaging errors across seeds, are conditional uncertainty, not independent
confirmation or15,430 independent samples. Motion-minus-quality is the primary
matched contrast. Positive flow amplitude alone is not predictive lift.

No deployment, primary-protocol change, Stage5C, SMC, seconds/metric/true3D or
foundation claim. A failed comparison does not prove that higher-resolution
perception or all visual information is useless.

## Pre-Fit Reader Amendment

The first real-data pilot stopped before constructing an optimizer or scoring
any predictions:51 admitted rows have zero past restoration radius and an
existing false support mask. These are not missing data to delete. The reader
now uses a harmless arithmetic denominator for those rows and zeros their
motion channels, retains their quality channels and preserves the original
zero-forecast support rule. No training budget, labels, split or endpoint was
changed. The initial registration is retained in Git4858403c; this amendment
is committed before the first successful training pilot.
