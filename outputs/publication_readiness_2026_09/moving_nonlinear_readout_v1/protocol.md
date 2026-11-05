# Nonlinear Conditional Risk Readout, Frozen Neural Representation

## Hypothesis And Fixed Intervention

The preceding affine output solve reduced fitting MSE but retained selected-set
risk error. This experiment tests whether the frozen representation contains
nonlinear conditional information useful for correcting that error. It is not
another last-layer optimizer, a new trajectory model or a multicalibration
guarantee. Existing failed monotone selected-set margins, coarse 16-cell
calibration, event heads and all-action forests remain separate negative tests.

Fit two matched conventional histogram-boosting residual readouts:

1. score_only: the two normalized signed risk predictions of the frozen SVD
   readout, without hidden features.
2. latent_score: the same two scores plus its 64-dimensional frozen neural
   hidden representation, with complete-query context retained.

Each arm predicts residuals for (H-.02R)/scale and (EH-.02ER)/scale, added to
the original affine predictions. It may change ordering, unlike a monotone
global safety margin. Labels and normalization are exactly those of the prior
mean head. Unknown outcomes retain their old envelope upper bounds and zero
lower reference; they are not removed or claimed observed.

Each of three fixed contexts has both arms, two scalar regressors per arm,
128 rounds per regressor, learning rate .05, at most 15 leaves, minimum 64
rows per leaf, no L2 penalty and no validation early stopping. Checkpoint every
eight rounds. Six primary fits, 1,536 scalar boosting rounds, zero new encoder
or Adam updates. This uses standard sklearn gradient boosting, not a new neural
world-model training claim. Local sklearn version is bound in checkpoints.

Retain original empirical query weights. For numerical Hessian stability, both
arms rescale active weights by their common mean, with zero L2 penalty. Actual
fitting MSE is reported in the original weight units, including the denominator
mass of zero-proposal queries. Hidden features, scores and proposal indices
are fixed before fitting targets are read. No source/record identifiers become
features. Same source-specific preprocessing and frozen encoder in both arms.

## Role Boundaries And Fixed Evaluation

Reuse exactly the three registered seed17 contexts, A=1 router, B=0 fitting,
V=2 held SOURCE readout from moving_readout_refit_v1. V is previously exposed
development, not an independent test. No new official data roles or H access.
All six fits freeze before any V readout. Model selection and threshold search
are prohibited; both arms and all contexts must be reported.

The original action bank, proposal router, reference-support head, two zero
signed-excess cutoffs, 2% risk budgets, easy-net check and support contract stay
unchanged. Decisions are computed before future labels/validity are accessed.
The same-count/amplitude controls remain NA if empty. Old 24-group primary
remains NA, not passed. No identity-fallback certification or retrospective pass.

Compare both arms to the frozen affine readout, and latent_score to score_only.
Report fitting versus held selected-risk residuals, benefits/harm of retained,
newly admitted and rejected proposals, original per-recording risk/support,
source-macro utility and missing-outcome lower/upper contrasts. Keep all seven
SOURCE units and 17 recording views. No CI or independent significance claim
from this fixed one-seed mechanism pilot. Native obs8/pred12 image pixels.

The fixed expansion condition requires latent_score's completion-lower utility
change versus the affine control to be positive in every registered context,
nonzero switching and no increase in known all/easy violations in each context.
It also requires positive source-macro completion-lower utility versus score_only.
This is a compute-allocation condition, not a replacement scientific gate.
Deployment remains unchanged regardless of pilot outcome.

## Runtime, Reproduction And Claim Boundaries

Native arm64, Torch CPU4/interop1, numerical thread pools limited to four,
single process, no workers/resource probing. Atomic block checkpoints, PID,
heartbeat, same-binding resume. Real-data pilot: both arms 8->16 rounds resumed
versus uninterrupted 16 rounds, exact trees and inference. First full fit also
has one independent complete refit. Full prediction/action/readout replay.
Local execution is appropriate; no duplicate remote jobs are submitted.

Standard histogram boosting supports weighted regression and warm start;
this implementation disables validation early stopping and checks actual
round counts. See the [official sklearn documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
No novelty or population-calibration guarantee is inferred from library use.

No raw data, private feature arrays or weights in Git. No metric, seconds,
human-gold, true3D, foundation, physical-safety or submission-ready claim.
Stage5C and SMC remain disabled. Slow healthy runs are allowed up to 12 hours;
only hard errors/resource limits interrupt the fixed experiment.
