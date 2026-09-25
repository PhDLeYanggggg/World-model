# Reference-Protected Risk Head

## Scope
This is a controlled continuation of a small Torch risk head, not a new
trajectory forecaster. It predicts four conditional cost moments for choosing
between an already fitted reference forecast and an alternative. The outputs
are all-population reference cost, positive harm, easy reference cost and easy
positive harm. Easy-event membership is a training/evaluation label, not an
inference input. The causal feature contract and delivered forecasts are fixed.

## Matched Arms
Both arms start from the same completed uniform mean head and receive 2,000
additional updates. Initialization, B-only preprocessing, loss scales, optimizer
reset, sample sequence and diagnostic batch are matched. The continued arm
updates the shared model. The protected arm obtains both reference moments
from a frozen copy and both harm moments from a trainable copy. The reference
copy has no gradients; its delivered C predictions are checked bit-for-bit.

There are 24,836 optimizer-registered parameters per arm, including unused
reference-output rows in the protected trainable copy. Protected storage is
49,672 parameters and inference uses two forwards. Equal updates do not imply
equal total storage, inference cost or effective-gradient parameter count.
Freezing predictions does not make their risk estimates calibrated.

## Training and Use
All six ordered source A/B assignments, seeds 17/29/43 and full/motion-only
forecast pairs are retained. A fits forecasts; B fits this head. C is excluded
from both current fitted chains, but was historically opened for development.
No C-based checkpoint or threshold selection is allowed. This is a diagnostic
research model; deployment is unchanged regardless of engineering completion.

## Safety and Limits
Greedy scene-query allocation is neither collision-aware nor a physical-safety
certificate. Unknown outcomes are not zero-error targets. Net easy degradation
and positive-harm constraints are distinct. A reference-only fallback is not
learned improvement. See results.md, failure_analysis.md and world_model_gate.md
for the completed study rather than inferring success from this architecture.

Obs8/pred12 annotation steps at raw stride12; image pixels, detector-derived
labels. No metric, seconds-level, human-gold, true3D, foundation or submission-
ready claim. No latent rollout; Stage5C and SMC are off.
