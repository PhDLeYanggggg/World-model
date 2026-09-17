# Observed Image Motion: Registered Fit-Only Comparison

The last two repairs failed:36 direct-RGB fits and45 objective controls did not
generalize across the three historically used fit scenes. The latter fit training
labels but introduced false motion on held stationary histories. The next test
is whether explicit, direction-aware past image motion adds usable observation
information beyond annotated geometry. This is a candidate-quality experiment,
not a threshold search or a claim that optical flow itself is novel.

## Locked Comparison

Retain all11,966 registered eight-observation/twelve-future-step windows, the
past-normalized ADE primary, equal physical-scene aggregation, original three
fit folds/seeds, and the existing easy definition. Development, calibration and
confirmation remain closed. Source labels upstream may be interpolated: this is
offline annotated forecasting, not strict real-time sensor causality.

Use fixed OpenCV4.13.0 Farneback, no pretrained weights. For seven consecutive
observed image pairs, restore96-to32 crop scaling and integer-center translation,
then use supplied row/column H only as a dataset-local coordinate proxy. Retain
forward/backward consistency and coverage as reliability proxies, not certified
identity, pose, physical flow, seconds or metric measurements. Center/ring medians
and their contrast are aligned to past heading and divided by the unchanged past
normalization scale. Zara03 retains every row with absent-motion masks.

Matched variants: quality_control (geometry plus common coverage/consistency),
magnitude (add unsigned motion quantiles), directed (also add vector medians and
center/ring contrast). All have identical input dimensions and network capacity.
The quality control itself contains flow consistency; it is not a pure no-motion
baseline. Report comparison to the prior geometry study separately, not as a
matched randomized experiment.

Two previously studied objectives are fixed before extraction/training: row_log
and scene_ade_harm. This prevents a negative motion result being attributed only
to logloss suppressing rare large errors, while retaining the known false-start
risk of ADE training. Three variants x two objectives x three seeds x three folds
=54 fits,4,000 updates each. No early selection, tuning or deployment promotion
from held fit scores; save the final fixed checkpoint and optimizer/RNG state.

Report paired exploratory three-scene bootstrap(2,000 draws), all/easy/stationary
and recording slices, fixed-predictor binary oracle and train-vs-held error.
Strong evidence would require positive held ADE, easy degradation<=2%, and a
directional gain over quality/magnitude controls. These are diagnostic fit folds,
not independent confirmation even if positive. Do not open sealed roles to rescue
a negative result. Synthetic flow/axes/crop tests precede fitting. Registration
binds implementation, inputs and this decision before any learned outcome.

## Remaining Scope

The intended contribution remains baseline-relative joint intervention under
scene-level risk control. Better observed motion would only enable that study.
Stage5C/SMC remain disabled. No primary/task/role change, new-data success claim,
or image/weight/raw-data upload is authorized by this experiment.

Implementation source:OpenCV's official dense-flow tutorial,
https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html.
