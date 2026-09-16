# Three-Seed Development Failure Analysis

Result source: `fresh_run`; completed real arm64 Torch training and development
evaluation. This is historically exposed ETH/UCY development data, not an
independent test, Stage37 revalidation or deployment certificate.

## What Ran

- Seeds 17, 29, 43, fixed before training. Per seed: full forecaster, three
  physical-scene-held forecasters, OOF ridge control and neural benefit/harm head.
- Each neural fit completed 1,000 updates: 15 fits, 15,000 updates. Training-only
  elapsed totals 880.61 seconds; preparation and evaluation are additional.
- 11,966 complete fit windows; 14,920 complete development trajectories and
  30,438 past-supported agent queries. Secondary endpoint FDE has 14,931 labels.
- Full eight-observed/twelve-predicted native-step paths. Raw-frame t+50 remains
  a separate, not-yet-evaluated supplement in this protocol.

## What Failed

The uncontrolled candidates worsen primary normalized ADE versus constant
velocity by 7.093%, 7.900%, and 7.249%. All seeds select the floor. Joint and
independent intervention give the same measured result at the tried policies;
there is no demonstrated coupling gain. The floor's zero change is abstention,
not a learned improvement or calibrated safety claim.

Even a future-label oracle choosing only between this candidate and the floor
has 0.722%, 0.704%, and 0.441% headroom on the primary metric. A weak or badly
scaled candidate cannot be repaired by threshold search alone. The oracle is
evaluation-only and never an input to the deployed decision.

Easy-case baseline ADE is close to zero. Relative degradation can exceed one
million percent for uncontrolled candidates; the JSON retains the absolute
errors and excess instead of suppressing these percentages. The denominator
problem is also a methodological limitation, not a reason to relax the frozen
2% criterion after seeing results.

## Competing Explanations

1. **Scale-sensitive training.** Fit-only audits show the top approximately 1%
   of ETH and Zara02 windows account for 96.32% and 99.74% of squared target
   energy. This is not a measured gradient fraction. Some small past scales
   produce very large normalized targets; a robust objective is a testable fix.
2. **Scale-sensitive evaluation.** Native-coordinate headroom is appreciably
   larger than primary normalized headroom. For seed 17, native ADE gain is
   -18.87% on Students01 and +0.60% on Students03. Metric sensitivity cannot
   be hidden or resolved by choosing the favorable metric post hoc.
3. **Limited generalization.** Development comprises one physical University
   scene. Three seeds show optimizer variation, not three independent sites.
   A meaningful physical-scene bootstrap interval is unavailable.
4. **Unproven mechanism.** There is no gain attributable to pairwise selection;
   public predictor, cost-sensitive deferral and actual-count-matched controls
   must still be run before a contribution claim.

## Next Registered Test

Use the same folds, rows, normalization, architecture, update budget, seeds,
cost-head loss and evaluation. Change only forecaster coordinate MSE to
Smooth-L1(beta=1). Retain the failed run and its code identity. No changing
test roles, deleting difficult records, new goals or post hoc safety thresholds.

Native-coordinate units and effective seconds remain unverified. Stage5C and
SMC remain disabled. This result does not meet submission-candidate quality.
