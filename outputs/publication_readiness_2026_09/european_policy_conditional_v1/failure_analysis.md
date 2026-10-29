# What Conditional Calibration Fixed and What It Did Not

## 1. It Is Not a Runtime or Missing-Fold Failure

All72 prescribed crossings completed. The source used for readout supplies no
labels to that calibrator. All24 source/context passports exclude the upstream
forecast producer and old controller. Head-fit and inner-held recordings are
disjoint. Old raw predictions and raw policies reproduce exactly. There is no
new independent-test claim: these original TRAIN recordings were already exposed.

## 2. The Rule Removes Useful Decisions as Well as Harm

Accounting partitions each raw decision into unsupported removal, penalty removal
or retention. It never changes a policy or uses labels as inference features.
Each view's masses are divided by its full known reference error, then repeated
views are averaged within locality and localities equally.

| Cohort | Occurrences | Known benefit pp | Known harm pp | Known net utility pp | Unknown envelope pp |
|---|---:|---:|---:|---:|---:|
| Raw selected |27,371|4.573691|1.336734|3.236957|2.354920|
| Removed for unsupported cell |6,039|0.339556|0.116199|0.223357|0.070623|
| Removed by conditional penalty |15,639|4.038390|1.214702|2.823689|1.529051|
| Retained |5,693|0.195745|0.005833|0.189911|0.755246|

Unknown envelope is a causal worst-case bound, NOT measured harm. The penalty
does suppress observed harm, but discards more observed benefit. Its negative
matched-count contrasts versus raw/global/support-only also show a worse
selected set at the same budget. This is not resolved by reporting lower MSE,
fewer interventions or a positive pooled easy margin.

## 3. Calibration Does Not Survive Subselection

There are109 retained cells with known labels across dependent views.23 have
positive observed easy-risk excess despite passing the corrected per-row rule.
Five received zero easy correction;18 received a positive correction.

For12/23, the target raw-cell residual already exceeds the correction learned
from other sources. This is measured failure of the fitted cell correction to
transfer; it does not uniquely identify whether source shift, sampling noise,
inadequate causal features or coarse conditioning caused it.

For the other11/23, the correction covers the target raw-cell average, but fails
after the within-cell action rule chooses a smaller subgroup. Thus the subgroup
that matters at deployment is not identified by a coarse cell average. No new
threshold was selected from these outcomes. The retained cohort has negative
mean easy excess overall, while individual cell/view failures remain: averaging
away local risk is not an acceptable repair.

These counts can be reconstructed from `failure_accounting.json`:
`retained_cell_accounting.details`, using `observed_easy_excess > 1e-12`, then
comparing `target_raw_easy_residual`, `applied_easy_offset` and
`target_retained_easy_residual`. The stored `predicted_easy_excess_pp` always
uses the conditional overlay on that cohort, including the raw cohort; it is
not the original uncalibrated score.

## 4. Unknown Outcomes Still Prevent Complete Support

The target retains101 unknown occurrences. The mean-locality unknown envelope
mass0.755246pp exceeds its known net utility0.189911pp. These aggregate numbers
do not imply all individual views fail:20/72 pass the full finite-completion
check.33/72 have undefined selected easy-risk denominators, including25 empty
policies. None is counted as safe. The budget and label mask stay unchanged.

## 5. What This Rejects

This rejects the registered fixed16-cell additive correction as a useful safe
repair of the rational head. It does not establish that every conditional risk
estimator is impossible. Nor does20 supported views prove a deployable method.
Earlier repeated selected-set/global calibration and recording-deletion consensus
also failed. Do not recycle those controls as a new experiment with different
thresholds, more rounds or renamed guards.

The next work should return to the cost representation and supervised target.
The concrete unresolved question is whether the **within-cell ordering of harm
relative to the benefit being sacrificed** can be learned from existing causal
history/rollout/quality information on recording-excluded data. Reconcile this
with prior capacity, quality-clipping and temporal-target negative results first.
Only a materially different signal or supervision mismatch justifies a new fit;
the present readout must not select its thresholds or a favorable source.

No independent roles, deployment, Stage5C or SMC changed. These are image-local
detector-silver raw-frame diagnostics, not physical safety or world-model success.
