# Frozen Task-Gradient Diagnostic

The matched auxiliary experiment learned membership but failed the strong
expected-cost gate. This source-development diagnostic asks whether shared
encoder interference is locally supported before another controlled repair.
Registration precedes calculation. Parent: a9d49e5c.

## Fixed Scope

All 288 frozen heads: six ordered source assignments, three seeds, full and
motion-only features, four held-locality fitting views, two matched arms.
Use each head's existing fixed fitting batch of 256 supported rows. Do not
sample a favorable new batch. Restore the exact fitting normalization and
loss scales. Future outcomes are fitting supervision only; held-locality
outcomes are not used for this diagnostic. Original producers, policies,
forecast errors, thresholds and independent roles remain frozen.

For the shared first linear layer, record BCE versus total cost, all-harm
and easy-harm gradient cosine/norm ratios. Four cost components retain
their original one-quarter MSE contribution after fitting RMS scaling.
Verify the four gradients sum to the total cost gradient. A zero gradient
has undefined cosine, not zero conflict.

Then clone the frozen model and its saved AdamW state twice. On the same
fitting batch, take one disposable cost-only step and one cost+BCE step,
using original clipping, weight decay, learning rate and momentum. Measure
cost and component changes and first-order projections of the actual update.
Discard both clones. No checkpoint is updated or new model deployed.
This is a local optimizer counterfactual, not held-out improvement.

## Interpretation Rule

Describe all views, pairs, arms and source assignments; dependent fitting
views are not independent samples. No significance test or new bootstrap
of overlapping batches. A strict majority of full-input auxiliary heads
must show both negative cost/BCE cosine (count separately) and worse
virtual cost with auxiliary than without to motivate one registered
cost-prioritized gradient repair. Numerical ties use max(1e-10,
1e-6 * pre-step component loss). This is a diagnosis trigger, not a
deployment gate, scientific risk budget, or proof of global causation.

If interference is unsupported, inspect label severity/context/support
rather than sweep task weights. Negative cosine alone is insufficient:
AdamW, clipping, nonlinear curvature and saved momentum matter. A fixed
final fitting batch cannot identify the entire training trajectory.

Prior work: Yu et al., [Gradient Surgery for Multi-Task Learning,
NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).
This diagnostic is not a new algorithm and makes no PCGrad safety claim.

## Claims And Runtime

Fresh gradient/virtual-step calculations, cached_verified parent checkpoints
and source data. No new held-out metrics, latent generation, SMC, policy
selection, risk calibration or confirmation access. Main forecast remains
8 observed / 12 predicted annotation steps, detector-derived pixel labels;
no metric/seconds, human gold, true 3D, foundation or physical safety claim.
Historical exposed Stage35/37 scores remain exploratory, not restored tests.
Native arm64 CPU4, interop1, workers0; resumable per-group outputs and
heartbeat. A successful replay validates the diagnostic, not the method.
