# Support-Fractional Harm Model Card

## Intended Use

A diagnostic loss intervention for source-development risk prediction. It
is not a trajectory predictor, calibrated event classifier or deployed policy.
No checkpoint is selected by current held results.

The original four-output EventMomentHead is unchanged: 383 causal features,
width64, 24,836 parameters, 2,000 AdamW updates at lr0.0003/wd0.0001, batch256,
gradient clip5. The output structure preserves nonnegative costs, all-harm
bounded by the causal disagreement envelope, and nested easy costs.

## Objective

The original training-RMS-normalized moment MSE is retained. On positive
disagreement support, an auxiliary learns the all/easy harm fraction of the
envelope using soft-target Bernoulli cross-entropy. Average within each sampled
locality and then across available localities, coefficient1 fixed in advance.
This fraction is not a probability that a harmful event occurs.

Prediction fractions used inside the logarithmic loss are limited to
[1e-7,1-1e-7] for finite arithmetic; inference still returns the original
bounded moment. Only numerical target overshoot up to1e-5 is tolerated and
counted. Genuine envelope violations stop rather than silently clipping harm.

The previous unconditional hurdle and pairwise-ranking losses are not renamed
as this intervention. This target includes supported zero-harm rows, does not
factor occurrence/severity, and does not rank realized H/(B+H). No claim of
architectural novelty, a new scoring rule or distribution-shift guarantee.

## Training/Inference Boundary

Three B localities determine preprocessing and the easy-event cut. The fourth
is excluded from fitting; the source-A forecast chain excludes these B
localities. Future outcomes are loss/evaluation labels only. No central
velocity, future endpoint input, test goals, or B-fitted selector mask is used.
The six ordered source assignments and three seeds are all retained.

Controls are cached-verified mean heads with identical initialization/draws/
steps. Zero auxiliary reproduces the original model exactly in regression
testing; interrupted/resumed training matches continuous training. Real
sampler counts/state and fixed-batch initial moment losses are matched.
Extra loss computation means equal optimization budget, not equal wall time.

## Limits

The 144 fits are repeated development views, not independent scenes. Their
train-fitted easy cuts differ. Primary MSE changes and tail/coverage guards
are only a gate for a subsequent policy experiment, never deployment approval.
No current C policy, independent calibration or confirmation is evaluated.
Failed or negative evidence remains in results and failure analysis.

Native arm64 Torch CPU4, interop1, workers0, optimizer/RNG checkpoints and
heartbeat. Private weights are not published to Git. Obs8/pred12 annotation
steps, raw stride12, image pixels, detector-derived labels. No metric/seconds,
human gold, physical safety, true3D or foundation claim. Stage5C and SMC off.
