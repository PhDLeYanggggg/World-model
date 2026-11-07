# Predict Observed Easy Reference Before Joint Repair

The previous causal joint edit selected no observed easy action in
its missing-support donor subset. Here predict an agent-level label:
the causal floor has positive observed ADE within the existing
easy cutoff and has at least one valid future point. Wholly unknown
rows supply no *observed* reference; this is not a claim that their
true error is large. This is an exposed development component test
on the unchanged native Moving A 8/12 annotation-step task.

Use the identical eligible rows for every score: not selected by
direct, at least one legal bank action, and a multi-agent query.
For training regenerate direct, predicted moments and old joint
features using the other six of the seven fitting sources for
each fitting source. Exclude all calibration and held sources.
Fit a fixed ridge and a shallow histogram gradient boosting
classifier using the same inputs: past velocity changes, path
length, recent motion, same-query distance and median motion,
causal scale and predicted reference/risk summaries. Weight
records equally within source and rows by inverse query crowd.
No hyperparameter search or probability calibration. Freeze
both learned scores before evaluation. The old controls are
maximal predicted easy-reference mass and maximal joint easy/safe
score across legal families, on the exact same agents.

Primary comparison: boosted top-one observed reference rate
versus both frozen controls across eligible queries, with
recordings averaged within seed/source, seeds averaged within
source and 3,000 paired source draws. Report first eligible query,
pooled record/query-weighted AUROC/AP, and descriptive first-query
hits in the original 35 missing-support views. Repeated views
and overlapping queries are not independent. Robust positive
top-one intervals versus both controls are needed before using
the classifier as a candidate for joint-policy calibration.

Separately use each learned score for the same eight-action
retrieval budget, preferring distinct rows. Order families for
a row by predicted easy excess, predicted gain, and stable
identity. Measure checked label-aware ordered repair capacity
only as a diagnostic, with the same scorer and no oracle
decisions exported. There is no operating policy test in this
experiment, no change to risk or support, no independent data,
metric/seconds claim, deployment, Stage5C, or SMC.
