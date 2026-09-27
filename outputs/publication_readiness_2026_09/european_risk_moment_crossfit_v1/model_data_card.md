# Diagnostic Model and Data Card

## Intended Use

Determine whether conditional CV-cost and positive-harm moments fit their own
source localities and survive a controller-source holdout. These are risk heads,
not new trajectory forecasters or a deployable selector. All144 heads use the
same64-wide,22914-parameter native Torch architecture. No winner is selected.

The frozen neural forecaster and fixed damping receive matched fitting budgets.
Each head predicts nonnegative reference ADE and envelope-bounded positive harm.
Reference error is unbounded softplus; the harm bound is the maximum distance
between two causal rollouts. This asymmetry is documented, not presumed to be
the established cause of failure. Two outputs share an encoder and squared loss.

## Data and Roles

Cached_verified European source-development geometry and frozen forecasts from
the parent seal. Only the twelve already opened source-training localities are
used. Producer training is disjoint from all four controller localities. Each
risk head fits three controller localities and predicts the fourth. The producer
and outer readout never provide head supervision; the outer readout is not
scored here. Independent selection, calibration and confirmation remain closed.

Inputs are355 causal geometry/rollout features. Future trajectory labels and masks
are used only to construct fitting cost targets and later compute diagnostic
errors. No future endpoint, target latent, central velocity, test goal endpoint
or held-source normalization. Zero-reference labels remain in the all-risk task;
unavailable labels are not imputed. Bins and constants are fitted on three sources.

Observed8/predicted12 at raw-frame stride12, image-local coordinates. Released
detector-derived silver tracks, not human-gold trajectories. Not metric, verified
seconds, physical safety, true3D or a foundation-model result. No new image/goal
or JEPA contribution is assessed. Locality bootstrap does not remove detector
error, development exposure, overlapping windows or shared-producer dependence.

## Limitations and Safety

The all-risk2% screen is diagnostic and omits utility/easy gating intentionally.
It is not the earlier full policy and cannot certify its safety. In-sample skill
versus inner-held skill can identify a generalization gap, not isolate one cause
without a matched intervention. Reference/harm calibration ratios with absent
support remain undefined. Smaller marginal MSE need not produce calibrated risk
after selection. No deployment change, Stage5C execution or SMC.

Weights and row predictions remain local and Git-ignored. Public artifacts are
protocol, code, aggregate outcomes and hashes. Cached replay is reproducibility
on existing assets, not a fresh raw-data reconstruction or an anonymous package.
