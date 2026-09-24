# Why Query-Pooled Net Risk Harms Exact-CV Cases

## Material Passport

2026-09-24; fresh post-decision diagnosis on cached-verified source predictions.
No new model, policy, threshold search, external readout or deployment. Four
development-exposed SDD sites; obs8/pred12 native annotation steps, stride 12,
annotation pixels. Future outcomes are grouping/evaluation labels, not features.

The corrected run verifies every decision-manifest and experiment identity,
raw past-coordinate row mapping, integer-frame and half-pixel coordinate contract.
All 175,756 histories have eight valid observations. The original pre-review
source and diagnostic are retained locally under
`data/stage_cvpr2027_experiments/zero_reference_support_prereview/`; its weaker
metadata checks are superseded, not presented as the verified implementation.

## Observed Mechanism

The previous net-population policies have nine harmed zero-CV query/seed
instances: three Transformer and six EqMotion. These are only four unique
windows from three tracks, not nine independent situations.

- None is stopped in the last observed step.
- None has an exactly constant-velocity eight-step past. A strict full-history
  CV veto cannot directly remove any of these failures, so it is not trained
  or deployed as their proposed fix.
- All nine fail the individual signed-risk inequality q_i <= .02*r_i. Their
  selection is enabled by query-level allocation, not pointwise permission.
- All six EqMotion instances share a query with selected negative predicted
  net risks. One Transformer instance also has this condition. Two Transformer
  instances have no selected negative risk; negative credits alone therefore
  cannot explain every case. The full-query denominator is another mechanism.
- Candidate ADE equals forecast disagreement on these exact-CV futures. The
  observed ADE is 1.85-2.59 pixels for Transformer and 3.97-6.18 pixels for
  EqMotion. Predicted net easy harm is much smaller; neither a mean estimate
  nor its feasible query sum guarantees individual protection.

These observations motivate, but do not prove, a causal model of failure. A
fixed recomputation must determine how changing credit and denominator rules
changes all decisions and accuracy, not merely remove identified bad rows.

## Support and Label Limits

`exposed_source_outcome_support_rows/tracks` counts other-site complete outcome
rows in the exposed source population. It is NOT the actual fitting draw count,
not rebuilt nested producer labels and not independent risk calibration.
Moving exact-CV events are rare. Outcome-conditioned group membership is never
passed to the policy. Complete futures and missing futures remain different
support regimes; a missing label is not a negative harm observation.

Raw histories contain released annotation coordinates, including generated
annotations where present. Past-only row access establishes causality relative
to the supplied annotations, not online provenance of their creation.

## Next Implemented Test

The separately registered risk-subsidy factorial compares signed versus clipped
expected net costs and full-query versus selected-only denominators using the
same frozen models. Selected agents can still share budget in the strongest
of these query-level controls. No arm is predeclared successful or safe.
All adverse outcomes, matching failures and missing support must be retained.

This is mechanism evidence, not a new best deployable model or independent
confirmation. DroneCrowd stays closed; Stage5C/SMC off. No metric, seconds-level,
true-3D, foundation or submission-readiness claim.
