# Cost-Aligned Positive Harm: Complete, Not Advanced

All 72 conditional cost heads completed with exact refits and inference replay.
The registered advance screen fails. The controlled loss change partly repairs
the Poisson head's intervention behavior, but does not establish a better cost
predictor or a deployable protected policy. No neural dynamics were trained.

## What Changed

We retained positive train-mean-preserving conditional H/EH predictions, seven
past-quality descriptors, fixed forest routing, raw B/R/ER, source recording
partitions, normalization, projection and the 2% selected-positive-harm budget.
Only the training objective changed from conditional Poisson deviance to the
registered squared raw signed-score surrogate. The equivalent targets retain
cross-terms with frozen benefit/reference predictions. No threshold search,
future filtering, exclusion of unknown labels or independent-role access occurred.

The original squared-loss pilot did not converge. Its exact-curvature amendment
preserved the objective and 1e-7 tolerance. In the completed experiment all 144
head-channel training losses decrease, the largest gradient is 9.999905e-8,
maximum iterations 20, and maximum TRAIN mean-preservation error 1.38e-14.
The remaining failure is not the earlier optimizer nonconvergence.

## Results

Counts aggregate repeated head-validation occurrences, not independent examples.
Complete support uses the existing unknown-completion reader and budget checks.

| Arm | Selected | Selected unknown | Complete support /72 | Known easy-risk violations | Upper-risk violations | Worst easy upper |
|---|---:|---:|---:|---:|---:|---:|
| Original forest |95,455|918|33|4|7|5.4058%|
| Additive harm control |112,456|1,143|19|20|42|1,200.1684%|
| Positive Poisson |111,031|1,050|37|2|11|18.0277%|
| Positive squared cost |96,720|926|36|4|7|5.7195%|

The large upper bounds are not observed degradation. Undefined/empty support
does not pass. The four known-label cost failures occur in112/124 and have
exactly the original forest's action hashes. The other three cost upper failures
occur in067; known risk is below 2% but unknown completion crosses the budget.

| Cost minus control | Signed-score MSE change, nominal95% CI | Full utility %, CI | Count-matched utility %, CI |
|---|---|---|---|
| Original |+0.104153 [0.018447, 0.221794]|+0.005740 [0.000781, 0.013143]|+0.002701 [0.000368, 0.005625]|
| Additive |+0.136300 [0.040365, 0.259004]|-0.280235 [-0.482063, -0.120791]|-0.061909 [-0.107857, -0.019212]|
| Poisson |-0.032008 [-0.071937, 0.007322]|-0.038081 [-0.065080, -0.017195]|-0.020938 [-0.040337, -0.006170]|

Utility is the conservative selected net-gain lower mass divided by full known
reference error mass, expressed as a percent. These are not ADE/FDE improvement
percentages. Count matching is within the same recording/frame query. Against
original, matched cost support is33 versus33, upper failures7 versus7 and worst
upper5.4196% versus5.4058%. The tiny positive utility contrast does not cancel
cost-error and worst-risk failures.

Intervals use 3,000 paired locality bootstrap resamples over12 already exposed
development localities. They are nominal, not adaptive-search-corrected or
independent confirmation. Three head seeds are not three end-to-end trainings.
The experiments hold recordings apart within localities, not localities apart.

## Interpretation and Next Discriminating Test

The changed loss reduces selected occurrences toward the original policy and
reduces Poisson upper-risk failures. However, MSE is worse than original in8/12
locality means, with the largest increases in110,067 and124. Improvement over
Poisson MSE is not supported by the nominal interval; utility is lower than both
learned controls. Thus changing the loss alone did not solve this fixed-leaf,
mean-preserving positive model's generalization problem.

The remaining causal explanations are not uniquely identified: sparse TRAIN
harm support, quality-feature distribution changes, an overly flexible positive
shape, and the gap between per-tree raw surrogate and projected ensemble loss.
The next experiment must separate these mechanisms on frozen TRAIN/development
predictions before another refit. In particular, inspect the three largest-error
localities and the unchanged known-failure cohorts in112/124, including zero-
harm TRAIN leaves. Do not infer which mechanism is causal from this post-hoc
locality table, filter rows by future labels, or repeat cutoff calibration.

## Compute and Provenance

- Scientific registration:6160fcea; transport-only recovery:2a2533d8.
- First invocation saved36 heads before an SSH checkpoint write stopped making
  progress. At diagnosis it had20h21m elapsed but31m37s CPU. That blocked time is
  preserved, not represented as training time or silently removed from history.
- Recovery hash-verified36 heads, fitted the remaining36, and finished exit0 in
  1,538.21s. Its fit/exact-refit time was1,429.02s, peak RSS9.427GB.
- Completed72-head fit/exact-refit time across invocations:2,997.06s, excluding
  pilot and the unsaved head attempt. This is not total wall-clock time.
- All72 remote checkpoints,152,728,232 bytes, were rehashed in the owned CREATE
  directory. No new local numerical cache, Slurm job or simulation job change.
- Recovery invocation:8,280 independent scalar checks. Final all-head verifier:
  7,541 scalar/bootstrap checks. Seventeen focused engineering tests passed;
  no claim that the entire historical test suite was rerun.
- Cost fits are fresh training across two invocations; resumed artifacts and
  three control arms are cached_verified. Transfer is not_run because the
  predeclared advance screen failed. Deployment is unchanged.

No independent selection/calibration/confirmation role was opened. The data
remain detector-silver, image-local obs8/pred12 at rawstride12. No metric,
seconds, physical-safety, true3D, foundation or submission-readiness claim.
Stage5C and SMC remain off.

Evidence: [summary](summary.json), [verification](verification.json),
[failure slices](failure_slices.md), [transport diagnosis](transport_recovery.md),
[Chinese reproduction guide](reproduction_zh.md).
