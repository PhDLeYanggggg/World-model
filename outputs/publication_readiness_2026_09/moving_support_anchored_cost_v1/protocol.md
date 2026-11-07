# Pricing Unobserved Outcomes In Matched Switch Decisions

The fixed-count seed-dispersion study improved mean cost fit but lost on the
original paired completion lower. Test a model-side repair on the same exposed
A data, with the same three actual EqMotion fold-2 seeds, all 125,540 rows,
same CV/forecaster paths, and same 3,225 switches per seed. No new forecasters.

Let `r` be the fixed single-seed control action and `a` the new binary action.
On a wholly unknown row, the maximum inter-forecast displacement `R` bounds
the absolute change in available-step ADE by the reverse triangle inequality.
It contributes only when `a != r`. With fitted probability `p` of a wholly
unknown future and frozen conditional cost `v = gain - positive_harm`, use
the training surrogate `(a-r)(1-p)v - |a-r|pR`. This is not the evaluation
metric, a guarantee, or an imputed position. After removing terms constant in
`a`, sort by `(1-p)v + (2r-1)pR` at the same fixed count per query. No coefficient
search. Independent additive decisions with a count constraint are not a joint
interaction model. The reference is a research control, not a certified policy.

Keep both parent cost banks frozen. Fit a support classifier per source and
forecast seed, with exactly the other 13 sources, using the parent causal
dispersion feature vector and equal-source/equal-query weights. Its only label
is whether any future step is available; held validity never enters inference.
Prespecified controls: original single-seed, original dispersion, visibility
weight only, constant-fitting-rate anchor, learned anchor. The constant anchor
separates a useful learned probability from generic inertia. Preserve ID tie
breaking, freeze all selections before any new held scoring, and reproduce
the original dispersion contrast.

Use the original held A easy/hard cuts and unchanged whole-12 available-step
gain, paired completion lower and known positive harm. Average across seeds
within source before the same 3,000-draw bootstrap. Keep every source and
recording, no defined-only primary. Primary: learned anchor against original
single-seed control. Both utility interval lower endpoint >0 and easy positive
harm increase interval upper endpoint <=0 are required to progress. Report
all controls and support Brier error, rather than promote the best one after
seeing the outcomes. Comparing each completion-lower bound to the reference
does not give a bound for comparing two treatments directly.

This follows regression-cost comparison rather than classifier accuracy
([Mao et al.](https://proceedings.mlr.press/v235/mao24d.html)). Related
abstention work studies baseline-relative uncertainty in policy learning
([Sawarni et al.](https://proceedings.mlr.press/v300/sawarni26a.html)); this
experiment instead prices disagreement between two fixed regression paths
when trajectory labels are absent. No claim to their guarantees or a general
novelty theorem.

Exposed development, not independent calibration or confirmation. Native
8/12 image coordinates. Same switch count is not the original risk budget.
No change to the original 2% all/easy, easy-net, or support/NA criteria, no
identity certification, no deployment change, no Stage5C or SMC.
