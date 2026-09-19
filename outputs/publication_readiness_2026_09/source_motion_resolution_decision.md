# Source Motion Resolution And Window Comparison

Registered before new decoding, flow extraction or fitting. This is a targeted
measurement/information experiment, not a new primary benchmark or deployment.

## Hypothesis And Population

The failed 32px box-flow comparison may suppress motion because downsampling
and a 45-video-pixel averaging window cover much of a typical annotated box.
Test both explanations, without assuming that either caused the failure.
Reuse all 15,430 stationary-history queries, 25,300 observed crops, 23,890
unique observed pairs, 29 recordings and four explored source sites. Preserve
51 unsupported restoration rows. Bookstore, main, outer and external readouts
stay closed. Approved offline supplied annotations, obs8/pred12, stride12 raw
frames; no sensor-as-of, metric, seconds, true-3D or foundation claim.

## Fixed Measurement Controls

Recover 96x96 retained past crops from the bound original videos. Their
supported 3x3 reduction must exactly reproduce every old RGB/coverage crop.
Four arms: lowpass_w45 (32px/winsize15), lowpass_w15 (32px/winsize5),
native_w45 (96px/winsize45), native_w15 (96px/winsize15). All remaining
Farneback parameters stay (.5,3,iterations3,poly_n5,poly_sigma1.2,flags0).
Restore crop translation and per-axis video/annotation resize only.

Nominal averaging-window extent is matched in video pixels. This is not a
perfect resolution-only contrast: sampling lattice, pyramid operation and the
pixel-defined polynomial neighborhood also change. Region boundaries use the
same continuous boxes, 3-video-pixel exclusion rim and 6-video-pixel outer rim.
Consistency tolerance is 4.5 video pixels, region support minimum144 video
pixel area. Native boundary exclusion3pixels versus lowpass1pixel retains the
old convention. Box/surround are proxies, not segmentation/camera-motion gold.
Lowpass_w45 must reproduce all 23,890 old flow pairs exactly.

OpenCV documents the robustness/smoothing tradeoff of averaging windows:
[official 4.13 reference](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html).
Prior ETH/Hotel/Zara native-flow results were negative; they are not silently
discarded or interchangeable with this SDD source-only subset.

## Fixed Information Test

For each arm, compare geometry+quality against geometry+quality+motion using
the unchanged 19-channel payload and training-only normalizers. Four held-site
cross-fits; logistic C1, no class weighting, lbfgs4000iterations/tolerance1e-8.
Two supervision-only labels from separately hash-verified exact raw future
coordinates: any nonzero displacement and maximum excursion strictly >10
annotation pixels (squared distance >100). The latter has728positives. Previous
739-positive float32-restored results remain historical, not the new control.
64 deterministic convex fits; no redundant seed repetition, threshold search
or hyperparameter selection. Coefficients and predictions checkpoint per fit.

Primary probe evidence: Brier and log loss, also compared with the training
prevalence constant. AUROC/AUPRC are complementary, not deployment evidence.
Report all resolution/window and motion/quality contrasts; 2000 paired site
bootstrap draws, seed17, four already-explored sites and shared training folds.
Intervals are conditional, unadjusted exploratory summaries, not independent
confirmation or a safety certificate. Do not count overlapping windows as
independent calibration data.

If higher-resolution motion has no consistent proper-score benefit, do not
spend another full trajectory-fit budget on the same representation. If it
does, register a separate matched trajectory experiment before fitting it;
probability lift alone is not trajectory, selector or world-model success.
No Stage5C, SMC, deployment change or altered primary protocol.
