# Failure Analysis

## 1. The Full Policy Still Fails

Primary fit coverage is 51/72 views and 160/444 recording views, worse than the
frozen full-gradient 53/72 and 178/444. Held coverage is 0/72 and 4/186. The new method
does not even solve every empirical fitting constraint. Generalization is
therefore not the only unresolved issue.

Known all-risk violations occur in 63/72 held views and known easy-risk violations
in 59/72. Missing outcomes matter, but cannot be the sole explanation. Of 57,618
held interventions, 902 have unknown targets. Those stay in inference and retain
completion bounds; the readout does not select only known outcomes.

## 2. Useful Choices Are Lost Along With Harmful Choices

On held data, primary known benefit/harm retention is 47.151%/46.610% relative to
origin. The known net change is -1.894111pp of reference cost. This is not the
desired selective rejection of harm. Both full and count-matched contrasts are
reported; reduced action count is not accepted as evidence of ranking improvement.

The same-record control has 59,385 actions versus 57,618 primary, yet its count-matched
contrast still favors same-record on the observed cohort. This separates the
point-estimate failure from the simple explanation of different switch rates.

## 3. Cross-Recording Training Has No Demonstrated Advantage

Held matched cross-minus-same completion bounds are [-0.188814,-0.044687]pp.
Seeds 17 and 43 have negative upper bounds; seed 29 spans zero. Sources 007 and 124 show
small positive matched bounds, but 110 and 119 show clear negative cohort contrasts.
All 12 sources fail whole-policy safety, including the locally positive slices.
No source or seed is dropped, promoted or used to retune the method.

The paired 12-source bootstrap outer envelope [-0.453565,+0.063353]pp spans zero.
This rules out claiming demonstrated improvement, but is not proof that every
cross-record meta-learning method is inferior. The inference is limited to the
fixed architecture, step size, objective and exposed source cohort.

## 4. Training Support Is Limited

Eight of 12 sources have a single fitting recording. Training episodes cannot put
that source on both sides without reusing the recording. All original recordings
were retained and the official split was unchanged, but those episodes mix
recording and source differences. Both sides also share fit-only preprocessing.
This is not a new calibration set or cross-fitted preprocessing. Existing
inner-held results remain historically exposed, not independent validation.

## 5. The Meta-Signal Is Small in Final Logged Objectives

At the final pre-update trace, 9/18 fits in each arm have exactly equal query and
unadapted hard-forward objective values. The median absolute difference is
4.470348e-8 for same-record and 2.607703e-7 for cross-record. Median outer gradient
norms are 0.004446 and 0.004040. These logged values do not prove equal gradients,
collapse or saturation: hard actions can stay identical while surrogate
derivatives differ. The second-order implementation passes independent dense
calculation and exact resume tests. A separate logit/gradient mechanism study
would be needed to attribute failure to the straight-through estimator.

No final inner side is clipped. Primary has 288 positive final pre-update
record/channel risk terms and 25/888 dual entries at the cap, versus 293 and 18/888
for same-record. These diagnostics are not final evaluation metrics and do not
identify the cap as the root cause.

## 6. The Next Discriminating Experiment

Before another architecture or meta-step search, measure whether each original
TRAIN recording admits a nonempty, useful decision set meeting the exact
approved constraints. A label-aware oracle would be diagnostic only, never an
inference feature or deployable result. Keep undefined and infeasible cases
visible. Where feasible actions exist, compare a hard-decision feasibility
repair against the current surrogate optimizer; where support is missing,
specify the data need rather than hiding the recording or changing the budget.

This follow-up has not run. No independent role, Stage5C or SMC is opened by this
diagnosis. The current result remains a failed policy-generalization repair,
not proof that neural world modeling itself is impossible.
