# Past Images Improve Fit, Not Reliable Intervention Risk

We study whether past-image context repairs an intervention controller whose
trajectory-only cost estimates fail under source shift. This is a registered
component experiment on previously exposed MovingDroneCrowd++ development
data, not independent confirmation. The native task observes8annotation frames
and predicts12in image coordinates. The complete62,466-window roster includes
partially and wholly unlabelled futures; missing outcomes are not assigned zero
error and are retained in the original completion-bound calculations.

## Matched Design

Four equal-capacity neural heads receive raw120causal trajectory/rollout features
with56context slots:zero image context, native image boundaries, measurement
quality with identity-suppressed background motion, or measured background
motion. This separates motion from reliability and redundant geometry. The
primary contrast is motion minus quality under the same joint-free allocator.
Image bounds minus no-image is secondary; motion minus no-image is a bundle.
These are not full image encoders or newly trained trajectory forecasters.

Training uses source-separated producer-A OOF donors only, not own-B outcomes.
Two producer supports, three risk-head seeds and four arms yield24models of
19,814parameters, each completing4,096updates. All24complete from-scratch refits
match the original scientific state exactly. A real128+128versus256recovery
pilot is exact. All36development decision contexts are frozen before outcome
readout. Reference/support heads see action-invariant context only. The original
2%all/easy positive harm, signed easy-net and reference/support conditions are
unchanged. Empty controls and absent references remain not estimable.

## Results

| Inputs | Known ADE gain vs CV | Supported original recording views | Worst signed easy loss |
|---|---:|---:|---:|
| No image | +0.7004% | 18/72 | 12.4108% |
| Bounds | +1.2611% | 26/72 | 9.5477% |
| Quality / geometry | +5.2899% | 12/72 | 47.3520% |
| Motion / geometry | +5.8670% | 12/72 | 118.2649% |

The registered paired completion-lower difference is+0.461541percentage points,
with exploratory source-cluster95%interval[-1.094911,+2.293574]. All three
risk-seed means are positive but three of eight source means are negative.
The primary screen fails. Secondary bounds and bundle intervals also cross
zero. Bootstrap3,000draws average risk seeds within each source before sampling;
overlapping windows and repeated risk seeds are not independent test units.
The72recording views represent24recordings, not72independent scenes.

Motion still causes43known-easy positive-harm violations and26signed-easy
violations; nine easy-reference views are not estimable. These loss ratios can
exceed100%and are not probabilities or physical safety risks. No model is
deployed. All36prediction banks and analysis scalars reproduce exactly in a
separate replay; frozen joint decisions satisfy predicted feasibility checks,
which do not establish realized risk calibration.

## Mechanism And Limitations

All six motion/quality matched pairs improve final training-minibatch loss.
On held sources, motion lowers observability Brier from.049945to.038765but
worsens harm MSE from.001455to.003445and easy-harm MSE from.000328to.000608in
common past-history-scale units. Gain/reference/easy-reference errors worsen
too. Better label-availability prediction does not repair forecast-harm learning.

The posthoc worst signed-easy recording case has a complete12-step future.
The quality control retains CV with ADE5.2922pixels; motion selects the full
neural forecast with ADE96.4910while predicting harm.3323. This is a selected
failure example, not representative sampling or a stand-alone pointwise budget
test. The existing trajectory actor remains frozen, so image context cannot
remove intrinsic candidate-forecast error by itself.

Background affine tracking is standard measurement, not a new vision method,
calibrated ego-motion, homography or metric reconstruction. A-only support also
differs from the old50%A/Baugmentation experiment; unmatched historical scores
are not pure modality contrasts. Three risk seeds are not three forecaster
seeds. No new C/H or reserved outcome readout is used. These exposed-development
intervals do not undo adaptive project history or establish generalization.

The experiment rejects this model-side repair as a deployment improvement under
the original contract. It does not prove that images are useless. Further work
must improve source-held conditional harm/reference learning or the underlying
forecast, rather than substitute mean gain for the failed original risk checks.
The compiled manuscript PDF is unchanged. No meters, seconds, true3D,
foundation, human-gold or submission-readiness claim is made. Stage5C execution
and SMC remain disabled.
