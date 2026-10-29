# Protected Risk Continuation: Research-Only Model

## Material Passport

A matched intervention on existing neural cost heads, not a new trajectory
forecaster, latent generator or independently validated world model. Thirty-six
real local continuations completed. Scientific results are in `conclusions.md`
and the immutable `summary.json`; this card describes the model contract.

## Inputs and Architecture

The existing 380 causal features include motion, neighbor and frozen rollout
diagnostics. Source means, standard deviations, scales and clipping are inherited
from the original inner-fit preprocessing. No future endpoint, observed future
mask, central velocity or test endpoint goal becomes an inference feature.

The original self-context head calls its copied width32 risk encoder on the
same causal vector twice and concatenates its representations. A two-output
bounded rational harm decoder predicts all-harm and easy-harm moments.
There are 12,322 trainable risk parameters and 25,471 total parameters. Benefit
and both reference outputs remain bit-exact relative to the original model.
This experiment adds neither peer pooling nor a new scene/goal modality.

## Learning Contrast

Six existing producer/controller contexts and head seeds17/29/43 produce18
origins. Each is continued twice from identical model weights, saved AdamW
moments, global step and source sampler state. Both arms use2,000 more updates,
four sources,16 queries per source, learning rate0.001, weight decay0.0001,
clip5 and final action temperature0.1.

The coupled arm retains moment/signed-score loss, task regret and source-scaled
dual action penalties. The estimation arm retains the identical primary
moment/signed-score loss only, freezing its unused duals. The contrast removes
both action-dependent terms together; it cannot separately attribute their
effects. Starting from action-trained weights also prevents a claim about the
best from-scratch estimator.

## Decisions and Support

The unchanged rule requires causal eligibility, positive estimated benefit
minus harm, estimated all-harm <=2% estimated reference and estimated easy-harm
<=2% estimated easy reference. These predicted inequalities do not prove
observed safety. Zero or undefined selected-reference support cannot pass.
Unknown outcomes remain in inference and support accounting; only known
outcomes supply loss targets and supervised MSE.

## Observed Result

Eligible all/easy MSE improved80.29%/55.65% over the frozen original, but fixed-
selection easy MSE did not improve. Only4/18 paired heads improved that target
against both controls, below the fixed12/18 criterion. Seven of nine fitting
conditions passed; the screen failed. Estimation selected419,270 repeated
occurrences with62/72 known easy-risk violations and0/72 full support. No held
readout or deployment promotion follows this result.

## Limits

All current data are exposed original TRAIN from twelve European localities.
The 72 fitting context/source/head-seed views overlap and are not independent
samples. Three cost-head seeds are not three independently trained forecasters.
Source preprocessing is not an unseen-domain adapter. The nine-condition
fitting screen is not a deployment gate and has no population safety guarantee.
Independent selection/calibration/confirmation and outer development are not
opened. Current observation/prediction remains8/12 at rawstride12, detector-silver,
pixel/local. No metric, seconds-level, human-gold, true3D or foundation claim.
Historical contaminated positive scores remain exploratory. Stage5C and SMC
are off. Checkpoints and data stay private; Git contains light evidence only.
