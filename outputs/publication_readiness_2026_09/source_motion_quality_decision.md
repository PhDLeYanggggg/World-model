# Fixed Source Motion-Quality and Past-Box Information Diagnostic

## Material Passport

Training-only follow-up to completed loss-control commit 414e4350. The original
candidate/CV oracle rose to 3.75968%, while actual equal-site gain fell to
-98.71920%. Rotated paths retain substantial oracle utility. This is a new
diagnostic registration after those results, not independent confirmation.

## Questions and Fixed Scope

1. How large and persistent are the nonzero annotation changes in the exact
   15,430-query, four-site stationary-history cohort?
2. Do its input/label provenance, generated coordinates and occlusion flags
   limit interpretation of these changes as actual movement?
3. Can observed neighbor directions or past box-shape changes predict direction
   or motion occurrence better than existing geometry alone?

No new primary metric, horizon, split, data admission, easy budget or deployment
threshold. All four sites and all rows remain. Bookstore and main/sealed roles
are not forecast or fitted. Raw annotations are read only for the 29 admitted
recordings contributing this cohort. Every sampled past/future center must
match the existing cache after its unchanged past-derived transformation.

## Descriptive Analysis

Use annotation-pixel maximum excursion bins: zero, (0,1], (1,2], (2,5], (5,10],
>10. Also normalize excursion by the median observed box diagonal, with edges
0,.01,.05,.1,.25,.5,1 and an open upper bin. No bin is called noise or true motion.
Report row/track/recording/site support, path length, endpoint/path ratio, return
to origin, first changed future step, final-four-step displacement, generated
and occluded flags, and history controls after query time. These flags diagnose
offline acquisition; they never become predictor features or row filters.

For each bin retain actual and binary future-oracle errors of both frozen model
families. Do not choose a policy or discard a poor bin. Check fixed past direction
hints: nearest moving selected neighbor, mean selected-neighbor velocity, and
toward the nearest current neighbor. Report support and endpoint cosine, with
undefined directions explicit. These associations do not establish usefulness
of a learned trajectory predictor or causal influence between people.

## Fixed Probability Probes

Four inner held-site folds, seeds17/29/43, two input arms and two labels: 48 fits.
Input arms: existing 480 observed-unit geometry/rotation columns; same plus38
scale/translation-invariant past box-shape features. Box features use eight
observed boxes only, no future boxes, identifiers, generated/occlusion flags or
track-remaining length. Geometry normalization is fitted on each complement.

Labels: any nonzero future coordinate change; maximum excursion >=0.5 median
past-box diagonal. The latter reuses the earlier source-support diagnostic,
not a new official task or an assertion of human-verified movement. No rows
are removed for either binary task.

Use ExtraTreesClassifier with128trees, max_depth8, min_samples_leaf64,
max_features0.5, no class reweighting, n_jobs4. Fixed seeds, no hyperparameter
or threshold search. Compare Brier error to the corresponding training-label
prevalence predictor; also report AUROC/AUPRC, calibration and support. Average
errors across seeds, not a deployed ensemble. Compare the two input arms by
2,000 shared physical-site bootstrap draws (seed38113); these are conditional
on four explored sites and shared fit populations, not independent confirmation.
Do not claim forecasting gain from a probability-only improvement.

Save each terminal forest and prediction receipt atomically, resume only
unfinished trials, and exactly replay all held probabilities. Loaded future-label
and raw future-box mutation must leave inference features unchanged. This does
not prove sensor-as-of availability of offline interpolated annotations.

## Decision and Resources

Small annotation magnitude alone will not justify changing labels or deleting
rows. If a new past feature provides stable probability lift, it motivates a
separately registered trajectory/cost comparison. If not, do not enlarge the
same selector merely to exploit oracle scores. No new policy or main evaluation.
The diagnostic is appropriate locally: raw annotations already exist, inputs
are small, and forests use four threads. Native arm64 environment, no worker
processes; trial-level resumable checkpoints and progress events. No raw data,
row caches, forests or images enter Git. CREATE state remains unverified unless
a fresh read-only connection succeeds. Stage5C execution and SMC stay disabled.
