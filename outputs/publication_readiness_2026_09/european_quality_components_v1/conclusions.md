# Harm Corrections Drive Unsafe Expansion

All72 frozen source heads and all13 registered component interventions completed
with exact inference/readout replay. This locates a failure mechanism; it does
not train, select or deploy a repaired model. Original and full-quality predictions,
actions and metrics exactly reproduce the previous experiment.

## Main Finding

The quality auxiliary adds22,805 and removes5,827 repeated action occurrences.
Of its additions,22,679 (99.45%) failed the old easy-risk condition,3,972 the old
all-risk condition, and951 nonpositive utility. These counts overlap. The main
opening mechanism is risk relaxation, not newly predicted positive benefit.

For added actions with a defined known-label easy reference, the equal-locality
mean observed easy-harm ratio is7.1334%, versus the unchanged2%budget. Its exact
decomposition is:

`7.1334% = 2% - 0.9154pp predicted excess + 6.2414pp harm underestimation - 0.1926pp reference inflation`.

This is62 defined head views in12 exposed localities, not62 independent samples.
It excludes299 added unknown-outcome occurrences, which remain separately bounded,
not zero harm. Harm underestimation dominates this average. Individual all-risk
failures can still involve reference inflation; this is not a universal cause.

## All Fixed Interventions

Utility contrasts are relative to original, in percent of full known reference
mass, not ADE/FDE gains. Complete support includes unknown completion, all/easy
selected risk, positive conservative utility and whole-easy preservation.
Undefined views are failures. Counts repeat rows across heads/controllers.

| Variant | Selected | Complete support /72 | Upper violations / defined | Known-label violations | Full utility delta (%) | Matched utility delta (%) |
|---|---:|---:|---:|---:|---:|---:|
| original | 95455 | 33 | 7/43 | 4 | 0.000000 | 0.000000 |
| quality | 112433 | 21 | 41/62 | 20 | 0.287244 | 0.061773 |
| only_benefit | 95288 | 29 | 6/36 | 3 | -0.000042 | 0.000010 |
| without_benefit | 112586 | 21 | 41/62 | 20 | 0.287380 | 0.061729 |
| only_harm | 98904 | 28 | 30/58 | 15 | 0.179582 | 0.016027 |
| without_harm | 106088 | 25 | 13/41 | 3 | 0.045466 | 0.025377 |
| only_reference | 95227 | 32 | 6/41 | 3 | -0.001104 | -0.000329 |
| without_reference | 112546 | 21 | 41/62 | 20 | 0.284147 | 0.060422 |
| only_easy_reference | 95794 | 30 | 9/44 | 5 | 0.000283 | -0.000142 |
| without_easy_reference | 112219 | 19 | 42/61 | 20 | 0.286838 | 0.063902 |
| only_easy_harm | 107974 | 32 | 16/50 | 7 | 0.048584 | 0.030648 |
| without_easy_harm | 99407 | 28 | 31/59 | 15 | 0.180864 | 0.015992 |
| quality_preprojection | 110037 | 16 | 32/48 | 12 | 0.151403 | 0.053143 |

Benefit-only and reference-only corrections barely change selection. Removing
harm correction lowers known-label violations20->3 and upper violations41->13,
but complete support remains25 versus original33. It is not a successful repair.
Easy-harm-only creates17,967 additions with16 upper violations. This component
also carries useful discrimination, but cannot certify safe conditional harm.

All nominal3,000-bootstrap locality intervals are retained in `summary.json`.
For example, removing harm still has matched utility+0.025377%,95%CI
[0.004159%,0.055302%], yet fails risk support. The previous complete quality
matched contrast+0.061773%[0.020429%,0.107359%] reproduces exactly.
Multiple diagnostic contrasts are not a policy-selection competition.

## Projection and Scope

The original feasibility projection adds2,456 and removes60 occurrences relative
to the nonnegative preprojection diagnostic. Coupling is material: capping total
harm can also cap easy harm. Disabling projection is not a fix: complete support
is16 and known-label violations12, and the raw moments can be infeasible.

Result provenance: `fresh_run` component inference, cohorts, arithmetic and
bootstrap; `cached_verified`72 original forests/quality checkpoints, row features,
splits and targets; `not_run`new training, policy selection, transfer, independent
selection/calibration/confirmation and full historical tests. Stage5C/SMC off.

Scope remains12 exposed European development localities,obs8/pred12 raw stride12,
image-local detector-silver. No metric/seconds, human-gold, physical safety,
true3D, foundation or submission-readiness claim. Deployment is unchanged.
