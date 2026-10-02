# Positive Harm Predictions Can Still Have Poor Conditional Cost Accuracy

Draft method and development-evidence section for the European detector-silver
track. This is not a replacement for the older SDD manuscript population, an
independent evaluation, or a submission-ready paper. Its scores must not be
pooled with historical SDD ADE/FDE tables.

## Setting and Estimands

We study whether a learned controller should replace a fixed causal forecast
with a fixed alternative. Future outcomes define five supervised moments:
positive benefit B, positive harm H, reference error R, reference error on the
easy subset ER, and positive harm on that subset EH. Easy membership is an
offline target definition, not an input available to the controller. The three
decision scores are B-H, H-0.02R, and EH-0.02ER. Prediction errors are normalized
by source-specific scales estimated on training data only.

The policy requires supported past inputs, predicted positive utility and
compliance with both predicted harm budgets. It does not follow that realized
risk satisfies those budgets. In particular, the evaluated easy-risk ratio is
selected positive easy harm divided by selected easy reference error. It is
not whole-easy net degradation: gains on other examples cannot subsidize this
ratio. Undefined denominators and insufficient support do not pass.

Unknown future outcomes are retained in the policy readout. For a supported
full-grid disagreement envelope between the two known forecast trajectories,
the triangle inequality bounds the possible change in their error. The existing
completion reader reports conservative unknown-outcome bounds separately from
known-label risk. A large completion upper bound is not a measured large harm;
neither may it be discarded when assessing the registered protection rule.

## Design and Evidence Roles

The cost experiments reuse fixed predictors and forest partitions on12 already
exposed development localities. Within each source locality, complete recordings
are separated for fitting and validation. The72 heads span fixed producer views
and three cost-head seeds. These are not72 independent datasets, and the three
head seeds are not three new end-to-end neural predictor trainings. Independent
selection, risk calibration and confirmation remain unopened.

All inputs are past-indexed. Seven auxiliary quality descriptors measure
history fit residual, finite-difference versus fitted-velocity disagreement,
observed width variation, reversal frequency, raw-prefix frame presence,
past detector confidence and partial neighbor support. Future coordinates,
future validity and offline easy labels do not enter inference. Missing labels
are not zero-valued supervision. Label sources remain detector-silver and
coordinates remain image-local under obs8/pred12 at rawstride12.

## Positive Conditional Harm Model

A previous additive quality regression improves cost accuracy but can produce
negative harm predictions, subsequently clipped to zero. We therefore test a
positive conditional head inside each fixed forest leaf. For standardized past
quality q and a known TRAIN leaf mean mu, the head is

```text
m_beta(q) = mu * exp(beta^T(q-q_bar))
                / E_train[exp(beta^T(q-q_bar))].
```

The denominator preserves the TRAIN leaf mean. Leaves with no observed training
harm remain at zero, without invented positive targets. Only H and EH are
modified; raw B, R and ER and all tree routing stay frozen. A shared feasibility
projection enforces the existing nonnegative, nested, disagreement-bounded
moments before applying the unchanged policy. These constraints ensure algebraic
consistency, not conditional risk calibration.

The completed positive-link experiment fits normalized conditional Poisson
deviance with a fixed coefficient penalty. It reduces known-label risk failures
from20 for the two-harm additive control to2, compared with4 for the original
forest. However, completion-upper failures remain11 versus7 original, and the
worst upper ratio is18.028% versus5.406%. Nine of the11 positive-arm upper
failures require unknown completion; two already fail on known labels.

## Frozen Error Attribution

We then freeze all models and actions and decompose the signed-score quadratic
error. Write the score transform as L, frozen score scales as sigma, and the
normalized errors of two moment predictions p0,p1 as e0,e1. The symmetric
contribution of moment j to the error difference is

```text
(p1_j-p0_j) * sum_s [L_sj * (e1_s+e0_s) / sigma_s] / 3.
```

Summing the five contributions recovers the score-error difference. This is an
algebraic identity, not a causal ablation. The reader preserves the historical
target-transform precision and sequential normalization. Unknown labels have
counts and bounds, but no fabricated MSE.

Across12 localities, the positive head increases final normalized signed-score
MSE by0.136161, nominal95% paired-locality bootstrap interval
[0.039763,0.257608]. EH contributes0.123727, or90.87% of this increase; its
interval is[0.034404,0.247105]. Frozen reference contributions are exactly zero.
Before feasibility projection, the increase is5.029890. Projection removes most
of that error, so this study does not support removing the projection.

Almost all excess error lies outside the selected set. On the final positive
model's selected cohort, the fixed prediction contrast improves global-weighted
MSE by0.000007310; on the complement it worsens by0.136168. This distinction is
important: global prediction error is not identical to selected-policy harm.
The small selected prediction error does not erase the observed risk failures.

Large multiplicative changes are associated with the failure. The EH ratio>4
slice contributes0.144035, more than the net increase because other slices
improve. A slice with low TRAIN effective EH support contributes0.123986.
These groups overlap and cannot be summed as independent causes. Label-based
overprediction strata are offline diagnostics only, not deployment filters.

## Controlled Loss Repair Being Tested

The diagnosis motivates a matched loss control, not a new post-hoc threshold.
We retain the same positive form, features, mean constraint, routing and budgets,
and replace relative deviance with a raw signed-score quadratic surrogate. For
fixed leaf B/R/ER and a=1/sigma_U^2, b=1/sigma_A^2, its equivalent supervised
targets are

```text
t_H  = y_H + [a*(B-y_B) + .02*b*(R-y_R)]/(a+b)
t_EH = y_EH + .02*(ER-y_ER).
```

Half squared error uses scales sqrt(1.5/(a+b)) and sqrt(1.5)*sigma_E. Its change
equals that of the three raw signed-score errors for fixed leaf B/R/ER. This
equivalence retains cross-terms with benefit/reference labels; simply fitting
H/EH squared error omits those terms. Effective regression targets may be
negative, but output harm remains nonnegative. This is a per-tree surrogate,
not exact optimization of the projected ensemble objective.

The first Gauss-Newton pilot failed stationarity even after a TRAIN-only2048-
iteration probe. Full analytic residual curvature solved the same failed tree
in7 iterations with the original1e-7 tolerance. A complete first-head pilot then
passed exact refitting and serialization checks. These are engineering findings.
The registered72-head experiment is running; no predictive improvement from the
new objective is claimed in this note.

## Limits on the Paper Claim

Nominal3000 locality bootstrap intervals describe development comparisons after
substantial prior inspection. They do not account for the full adaptive search,
establish exchangeable calibration samples, or certify future risk. Replaying
checkpoints and independent arithmetic verifies implementation, not independent
research replication. The positive-link and squared-loss changes are standard
modeling controls; their construction alone is not a novel world-model method.

A substantive contribution still requires positive controlled intervention
results, support for unknown outcomes, applicable independent risk calibration,
held-out confirmation, and matched strong forecasters. These studies establish
neither scene-joint superiority nor new neural dynamics. No metric, seconds,
physical-safety, true3D or foundation claim is made. Stage5C and SMC remain off.

Evidence: [positive training](../european_positive_harm_v1/verification.json),
[frozen diagnosis](verification.json), [loss protocol](../european_cost_aligned_positive_harm_v1/protocol.md),
[solver amendment](../european_cost_harm_newton_v1/protocol.md).
