# Past Appearance Did Not Repair Cross-Scene Forecasting

## What Was Actually Run

`fresh_run`: 18 real PyTorch models, three seeds (17/29/43), two held-fit-scene
directions and three input arms, each at the registered 1,000-update budget.
The pilot's 100 updates resumed rather than restarting. Total completed budget:
18,000 optimizer updates; summed fit time 135.13 seconds on native arm64 CPU,
four compute threads, one interop thread, no DataLoader workers. This is a small
diagnostic experiment, not medium/full training or a final benchmark result.

`cached_verified`: all 18 final checkpoints reproduce saved held predictions,
probabilities and guarded predictions exactly (maximum difference 0). A second
resume invocation verifies all 18 receipts with zero new training/evaluation and
preserves the original completion receipt. Four focused tests passed, including
real optimizer/RNG resume and modality-mask isolation; no legacy-suite pass is
claimed.

We retain all 365 frozen stationary windows: 81 ETH and 284 Hotel, only 31 agents
and 45 stationary runs. ETH has five agents, Hotel 26. These are overlapping,
historically inspected fit data, not independent evaluation scenes. No changed
split, primary metric, threshold selection or final-test access follows.

The input cache contains 2,920 requested past images at 458 unique native frame
indices. Of those requests, 2,664 have complete centered patches. All eight
patches are available for 333 rows; the other 32 remain present with masks and
exact zero/CV fallback under the fixed gate. No future image or target is an
input. Future displacement/change labels are loaded separately for loss/eval.

## Matched Predictive Comparison

All arms use the same causal geometry, supplied-H output mapping, 16,153-parameter
module shape, optimizer and update budget. The geometry arm does not exercise
the image encoder; current-RGB encodes one image; past-RGB encodes eight ordered
images. Effective capacity and compute are therefore not matched. A classifier
score of 0.9 is a fixed diagnostic gate, not a calibrated safety guarantee.

Three-seed means below are ADE improvement against zero/CV on the stationary
subset. Positive means better; negative means worse. Native and unchanged
parent-normalized ratios agree because this subset has the same scale floor.

| Held scene | Input | Unrestricted gain | Fixed-gate gain | Switch rate |
| --- | --- | ---: | ---: | ---: |
| ETH | Geometry | -0.39% | -0.01% | 1.23% |
| ETH | Current RGB | -7.69% | -8.09% | 65.84% |
| ETH | Eight past RGB | -14.53% | -14.98% | 66.67% |
| Hotel | Geometry | -513.48% | -132.04% | 20.54% |
| Hotel | Current RGB | -389.04% | -167.73% | 35.21% |
| Hotel | Eight past RGB | -401.80% | -172.63% | 35.68% |

None of the 18 guarded settings has positive gain. Adding RGB reduces some
unrestricted Hotel damage relative to geometry, but it does not beat CV and the
guarded outcome is worse. No seed or input arm is promoted. Still/easy CV error
is exactly zero, so percentage easy degradation is undefined. Absolute easy
harm is reported in [every-setting metrics](metrics.json), not counted as a
passing <=2% gate.

## Diagnosed Failure, Not Proven Causality

The post-fit [diagnosis](diagnosis.json) replays the same final models on their
own training rows without fitting or selecting anything new.

1. **Very small and asymmetric support.** The ETH-to-Hotel direction trains on
   only five stationary agents. Train annotation-change rates differ: 72.84%
   ETH versus 45.42% Hotel. A large row count would misrepresent this support.
2. **Training fit does not transfer.** ETH-trained geometry/current/past models
   gain 65.09%/68.95%/68.30% on their own training rows, yet lose
   513.48%/389.04%/401.80% on Hotel. This supports an overfitting/distribution-shift
   diagnosis, not a universal inability to optimize the network. In the reverse
   direction, Hotel training trajectory gains are already negative
   (-3.18%/-2.50%/-3.34%), despite positive in-sample start Brier lift.
3. **Camera/geometry support mismatch.** Under ETH-only normalization, 78.52%
   of Hotel rows have at least one feature outside the fixed +/-10 range.
   The image-x-to-local-normal Jacobian clips on 67.61% of Hotel rows; circle
   indicators/radii are unseen constants in the ETH training subset and clip
   on 40.14%. Hotel-to-ETH clips no rows. This is measured covariate shift,
   not evidence that clipping alone caused the forecast failure.
4. **Probability confidence is not trajectory utility.** All held-scene start
   Brier lifts are negative while all training Brier lifts are positive. RGB
   models switch far more often on ETH and damage previously exact still cases.
   Predicting that an annotation will change does not identify a useful direction
   or establish expected gain relative to CV.
5. **This appearance representation is weakly supported.** Fixed 96px crops
   resized to 32px, with no verified body/pose labels or pretraining, can retain
   camera background and occlusion. Eight frames do not beat one here. This does
   not show that visual motion/pose is intrinsically uninformative; it rejects
   this small cross-scene predictor under the stated assumptions and budget.

## Consequence for the Research Route

No new deployment or submission-quality contribution is established. This
experiment closes an actual predictive ablation after the media-access audits;
it is not a reason to retune confidence on the same outcomes. The clean parent
comparison still selects CV. Historical Stage37 gains remain exploratory under
the lineage audit, not an independently confirmed floor for this new protocol.

The next justified repair is a predeclared camera/support treatment tested with
the same frozen predictors first, followed by a genuinely broader-scene training
study only if it supplies predictive evidence. A start classifier alone must
not control trajectory intervention: utility/harm need their own independently
evaluated target and held calibration support. Additional model scale or
another threshold sweep on five training agents is not the priority.

The user-approved task remains eight observed / twelve predicted native steps,
with raw-frame t+50 supplementary. A prospective primary-metric change remains
pending, not silently adopted. Independent scene support, source-use scope and
the formal evaluation protocol still constrain a publication claim. None of
these observations certifies physical time, meters, pose, human-gold labels,
true 3D, foundation-model capability, Stage5C or SMC. The research goal remains
active and incomplete.
