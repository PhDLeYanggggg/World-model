# Longer Training Improves Average Fitting, Not Selected-Risk Reliability

## Completed Experiment

All18 original interaction-feature cost heads finish their fixed fourfold
budgets with actual seeds17/29/43:232,320 canonical primary updates plus an
8,000-update scratch refit of the first head. Every original-budget optimizer,
sampler, RNG, preprocessing and parameter prefix matches exactly. A separate
18-head fitting replay reproduces every decomposition. No new H policy score
or threshold search is part of this experiment.

The original disk-guard failure is retained in [interruption.md](interruption.md).
The resumed process exits0 in570.670s, peak RSS12.07GB; this excludes the first
process time. The separate replay exits0 in182.927s, peak RSS12.39GB. Unsaved
updates lost at the original interruption are unknown within zero to99 and
are not counted as preserved/canonical updates. No budget, role or reserve
was reduced. All26 completed-backup files,12,337,053bytes, were read back and
hash-verified on CREATE; this is storage preservation, not remote training.
The earlier interrupted backup remains separate and unchanged.

## What Changed

The equal-job full-bound fitting objective falls from1.326343 to0.865164
(-34.77%). Its observed-only diagnostic objective falls39.89%. These are
fitting objectives on the same models, not two new independently trained arms.

On exactly the original selected rows, amplitudes and actual paths, median
predicted/target easy-harm mass rises from0.003209 to0.009333. It remains below
one percent of the conservative/observed evaluated target. Falsely-safe easy
views on this fixed set fall from324 to115: the model recognizes more of the
old unsafe interventions, but the unchanged paths still have324 easy-risk
violations. The general harm ratio is0.3254, versus0.3315 before; better total
loss has not improved every risk component. Easy-reference ratio falls from
0.05994 to0.03807 and also remains poorly fitted.

When allowed to choose its own fitting interventions, the extended model
selects573,770 repeated row/pipeline instances instead of672,807. Median
easy-harm ratio falls to0.002782, with307 easy-risk violations; all307 are
again predicted safe and underpredict the easy-harm numerator. There are
147 all-risk violations. These are444 repeated recording/pipeline views,
not444 independent scenes, and one own-selection easy denominator is undefined.
Less intervention and fewer violations do not establish the original complete
risk contract, held generalization or deployment eligibility.

## Interpretation And Next Action

Fourfold exposure helps average regression and identifies some previously
mispriced old decisions, but it does not repair the residual optimism on
decisions selected by the new model. This rules out a simple claim that the
current failure is solved by this amount of additional training. It does not
prove an irreducible information ceiling, a unique loss defect or useless
interaction features. No unbounded training-duration sweep follows this result.

The next descriptive G-only diagnosis partitions these same fixed target masses
into complete labels, partial labels and wholly-unknown-label bounds. It will
determine how much of the remaining numerator is measured available-label harm
versus conservative missing-label contribution before choosing a model or data
repair. Observed-only subsets cannot replace the original risk contract.

The original feature primary remains inconclusive and its risk gate failed.
SOURCE remains exposed development; independent calibration was historically
failed, not untouched. No calibration/confirmation/DroneCrowd outcome is opened
for this repair. Main v9 PDF is unchanged; the new evidence is an addendum.
No new trajectory dynamics, metric/seconds/human-gold evidence, deployment,
submission-readiness, Stage5C execution or SMC claim follows.
