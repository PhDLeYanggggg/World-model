# Fixed Four-Source Producers With Disjoint Controller Roles

Registered 2026-09-25 before new fitting. This is a development-only repair,
not a new primary split or independent confirmation. Push code/config/tests and
this protocol before the first optimization update.

## Hypothesis

The preceding producer-tag experiment exposed two problems: tag effects change
sign across bundles, and two-source floors already violate the easy criterion.
This study keeps the original four-source candidate and floor fixed. It tests
whether a controller trained on forecasts from that exact producer transfers
better than a capacity/budget-matched controller trained on the old OOF producer
chain, when both operate on identical final forecasts.

## Roles And Common Event

Use the existing three disjoint groups of four opened EuropeanSquares source
localities. For every ordered A/B pair, A is producer fitting, B is controller
fitting, and the remaining group C is readout. All six directions, seeds17/29/43
and all/easy event targets are required:36 groups. Reserved model-selection,
calibration and confirmation roles stay closed. A C-source in one development
rotation can be A/B in another; this is not independent confirmation.

The original producer-A-derived easy/hard cutoffs are unchanged and apply to
both training arms and the common C readout. They use no B/C statistics. Each
arm learns feature normalization, error scale and ranking denominator from B
only. Candidates/floors/preprocessors used to score B and C exclude those source
labels. B's existing OOF controls exclude the source being scored within B and
all C sources. Predictor-trained A sources are never used for controller losses.

## Training Arms

1. Producer-matched: B features and gain/harm labels use A's four-source neural
   candidate and protected motion floor. C uses the exact same A bundle.
2. OOF control: B features and labels use B's existing opposite-half two-source
   producers/floors. C still uses the exact same A four-source bundle as arm1.

Both arms are newly fitted:380 features, width64,2,000 fixed final updates,
batch256, same initialization seeds and source-balanced supervised draws, same
optimization settings and2% predicted risk budget. Training rollouts and their
cost labels deliberately differ; preprocessing/ranking scales can consequently
differ. Thus neither label matching nor uniquely causal attribution is claimed.
The new event labels for both arms use A's unchanged cutoff, not B's old cutoff.

144 Torch heads,288,000 updates. The first100-step utility/risk pilot resumes
inside the budget. Some OOF utility fits repeat identical supervision across A
rotations; these are paired engineering controls, not independent replications.
Also fit72 fixed-alpha .01 source-weighted ridge moment heads on matched B
features as a simple baseline. Nonnegative/causal-envelope projection is fixed.
No alpha, threshold, checkpoint, branch or event is selected from readout.

## Readout

Freeze every head and all180 decisions before new comparison readout. Policies:
producer-matched, OOF-control, ridge, old stopping-protected controller, and raw
neural diagnostic. The unchanged A four-source protected floor is the common
reference. All neural controller interventions use latest-observed-step stopping
protection, predicted benefit above harm, and predicted risk<=.02*reference.
Raw neural is an explicitly unprotected diagnostic, not a deployment candidate.

Primary comparisons are matched versus OOF-control/ridge/old-stop on identical
A forecasts and C rows. Report all/easy/hard/complete ADE, endpoint FDE,
positive-easy degradation versus CV, zero-CV absolute harm, unknown-label switches,
worst-locality/tails, selected positive-harm reliability and intervention rates.
Use3,000 paired locality bootstraps, seed39271, with exactly four expected C
localities per view. Retain every rotation; do not pool overlapping rows as
independent observations. Counts across36 dependent views are descriptive.

Verify source exclusion, native checkpoint replay, identical supervised draws
between neural arms, all frozen choices by scalar replay, native coordinates and
aggregate metrics by separate arithmetic. Preserve old stop decisions exactly.

## Limits And Promotion

This manipulates the source/size/normalizer/quality of training forecasts jointly,
not producer identity in isolation. Old-stop versus new-head differences also
include which sources supervised the cost head. A fixed producer does not remove
B-to-C covariate shift or calibrate risk. Positive mean gain is insufficient if
worst-locality easy degradation exceeds2% or observed zero-reference harm appears.

There is no new trajectory-forecaster training, independent calibration, deployment
promotion, Stage5C execution or SMC. Image-pixel obs8/pred12 at raw annotation
stride12, detector tracks, not metric/seconds/human gold/physical safety/true3D/
foundation evidence. Unknown future labels are not zero-error observations.
