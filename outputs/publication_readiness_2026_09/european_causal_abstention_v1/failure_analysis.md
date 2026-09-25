# Failure Taxonomy

## 1. The Motion Guard Used the Wrong Past-State Question

The prior guard asked whether there was any movement during eight observations.
It admitted trajectories that had just stopped. Fresh stopping protection removes
the observed zero-CV harm without any future input. Its all-ADE change is negligible
and no paired interval excludes zero. The correction is to the decision contract,
not to a learned neural dynamics law.

Four unique rows, one locality, two valid future labels per row, no endpoint.
Both folds evaluating those rows have no zero-CV fitting examples. In addition,
the positive-easy objective excludes zero-CV labels by definition; the all-event
objective includes known labels but cannot learn an unobserved fitting case.
This does not establish that stationary agents stay still for the full horizon.

## 2. Feature Support Is Not Predictive Value

The support filter removes more useful than harmful interventions under the same
per-locality floor denominator in every parent/view. For CV targets, removed
all-benefit ranges from 0.0224 to 0.1435 percentage points, versus avoided harm of
0.0030 to 0.0256 pp. These are ranges over different views; the per-view ledger,
not subtraction of range endpoints, establishes the negative net effect.
For floor targets, the corresponding ranges are 0.0202 to 0.1364 and 0.0034 to
0.0727 pp; net effect again negative in every view.

An axis-aligned 1%-99% box accepts a central part of a feature distribution, not
the part where replacing the floor is beneficial. This observed objective mismatch
is supported by error accounting. Whether a learned density estimator, different
representation or additional data would fix it is not established.

## 3. Some Nominal Ablations Are Degenerate

Stop and both count controls are identical in all 72 parent/group combinations:
there is no cohort with a positive quota smaller than the original intervention
count after stopping protection. This dataset slice cannot distinguish ordering
methods for that guard. Support does admit flexible cohorts: 3-3,439 for CV targets
and 2-2,788 for floor targets per view. Counts are current-frame cohorts, not
independent scenes or training examples.

Support and combined policies are identical. Each fold has four moving-state
source boxes but zero or one terminally-stopped source box, below the two-source
requirement. Counting their duplicate scores as corroborating ablations would be
misleading. Every registered arm remains available in the result archive.

## 4. Easy Ordering and Overall Utility Trade Off

At equal intervention counts, support sometimes protects positive-easy labels
better than risk/random ranking. This is real conditional evidence of an allocation
tradeoff, not a uniform failure on every metric. However, all-ADE and hard ordering
are inconsistent, and total utility is below the unchanged parent in every view.
Neither the easy-event limit nor the few zero-CV repairs justify selecting this
policy for deployment from the same opened results.

## 5. Producer Transport Remains Unresolved

The support vector includes neural-to-floor rollout disagreement. Fitting rows use
source-cross-fitted producers trained on two localities, while held development
rows use frozen four-source producers. Thus rejection may reflect a change in the
prediction-producing system, not only unusual observed motion. This remains a
plausible confound, not a proven explanation for all loss. The next comparison
should separate pure-history support from producer-dependent disagreement while
keeping costs, cohorts and controls fixed, not tune percentile thresholds.

## 6. Confirmation, Labels and Physical Claims

These source-excluded development folds are not independent confirmation after
iterative research. The overlapping 318,969 indexed rows contain only twelve
localities. The locality bootstrap is conditional and is not a conformal or
physical-safety guarantee. Released detector tracks and partial labels are not
human gold. Motion in image pixels and raw annotation steps is not metric/seconds.
No independent interaction-conflict improvement was evaluated in this guard study.
No new dynamics model, deployment, Stage5C execution or SMC was introduced.
