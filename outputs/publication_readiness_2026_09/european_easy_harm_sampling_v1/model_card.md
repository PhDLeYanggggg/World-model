# Risk-Head Sampling Model Card

## Scope
These are 36 experimental cost heads, not new trajectory generators. The
existing full and motion-only forecast pairs and utility scores are fixed.
The intervention controller chooses between delivered R and P forecasts.
No deployment is selected from this study.

Each native PyTorch head has 383 causal input features, one width-64 GELU
hidden layer, four coherent nonnegative outputs and 24,836 parameters. Outputs
are all-reference mass, all-positive harm, easy-reference mass and easy-positive
harm. Easy outputs nest within all outputs; all harm is bounded by the causal
rollout-difference envelope. A bounded prediction is not a calibrated upper
confidence bound.

## Fitting
Every head trains 2,000 AdamW updates, batch 256, learning rate 0.0003, weight
decay 0.0001 and gradient clipping at 5. Seeds are 17, 29 and 43. B-only
preprocessing, loss scales, initialization and fixed diagnostic batches match
the preceding uniform mean-loss controls. The corrected draw stream differs
by design. There is no architecture, objective, tolerance or forecast change.

Within each B locality, half the sampling mass is uniform over supported rows
and half proportional to supervised easy harm. Multiplying loss by p/q
preserves its expectation. The ratio is at most two without clipping. Future
labels are permitted in this training sampler and loss, not in inference.
Unknown labels are excluded from fitting; their availability never gates an
inference decision. The loss identity does not imply unbiased Adam updates or
improved calibration. A 100-step pilot resumes inside the first 2,000 updates.

## Evaluation and Limitations
All 504 decisions were frozen before C outcome readout; nine old control
policies per group are cached_verified and five are newly evaluated. No
test-selected threshold, seed, checkpoint or fitted posthoc correction is used.
Source C excludes current A and B fitting chains but has historical development
exposure. It is not independent confirmation. See results.md and
failure_analysis.md for every source assignment, including negative outcomes.

The greedy scene-query rule is not optimal, collision-aware or physically
certified. Whole-population risk and policy-selected conditional risk differ.
The number of training windows is not the number of independent scenes.
Importance sampling itself is established prior work, not this project's
novelty claim. No new neural-dynamics lift is established by changing a risk
head's sampler.

Units are image pixels and annotation steps (obs8/pred12, raw stride12), with
detector-derived labels. This is not metric, seconds-level, human gold, true
3D or a foundation model. Stage5C and SMC are off. Reserved calibration and
confirmation remain closed.
