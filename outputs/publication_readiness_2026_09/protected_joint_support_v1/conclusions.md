# Protected Joint Choices: Real but Extremely Sparse Opportunity

## Evidence Status

2026-09-22. `fresh_run`: exhaustive causal support computation and exact replay.
`cached_verified`: frozen ramp forecasts, forest/log/square risk scores, nested
producer lineage and identity-resolved past context. `not_run`: new training,
new ADE/FDE readout, outcome-driven policy selection, calibration or deployment.
Registration was committed and pushed as `188170f7` before this computation.

The audit covers all 33 recordings and 20,932 recording/frame groups at three
seeds: 62,796 repeated scene/seed queries for each pool. There are four physical
sites, already development-exposed, not 62,796 independent scenes. This remains
an 8-observed/12-predicted annotation-step, pixel-space study.

## Fixed Question

The earlier scene-joint control produced exactly the same choices as its unary
control. Do the newer proposal pools offer more opportunity for agent-pair terms
to change a decision? The purpose is to check prerequisites before another joint
model, not to assume that changing decisions improves real future trajectories.

Three fixed pools use identical ramp forecasts: forest relative-risk (identical
to its strict policy), old log strict and new square strict. The inherited
half-count rule, reference-predicted-harm budget, train-only cost scale, past
neighborhood radius, proximity fraction 0.1 and pair weight 1 are unchanged.
Full-count selection is still structurally unique. At half-count, products
cannot contribute unless at least two eligible agents can be selected together.

## Results

| Fixed pool | Eligible query/seed instances | Queries with half-count >=2 | Non-additive opportunities | Changed joint-vs-unary queries | Unique changed recording/frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forest relative-risk | 22,539 | 658 | 62 | 3 | 1 |
| Old log strict | 33,793 | 1,143 | 91 | 4 | 4 |
| New square strict | 37,030 | 1,391 | 113 | 3 | 3 |

All **266 pool/query opportunities** were exhaustively checked: **61,024**
candidate subsets and **18,521** feasible subsets. None hit the enumeration cap;
the largest set had only 6,435 combinations. This is a complete calculation for
the registered pools, not a quick subsample. Direct pair-sum and decomposed
objectives agree for every enumerated feasible subset.

The forest's three changes are all **hyang/video0, annotation frame 11184**, one
instance per seed. Their predicted objective advantages are about 0.00324,
0.00315 and 0.00328. They are three repeated model evaluations of one scene
frame, not three independent interaction events. Neither coupa nor gates has
any forest-supported non-additive opportunity under this fixed rule.

Old-log changes involve four unique frames; new-square changes involve three.
Across all pools, the ten changed query/seed instances cover only six distinct
recording/frames. All changes improve the designed predicted objective beyond
the numerical tolerance. **Actual ADE, easy-case error and collision outcomes
for these changed choices were not evaluated.**

Most potential couplings are too weak to change the unary order: the entire
feasible product-cost range is below the unary runner-up gap in 56/62 forest,
66/91 log and 84/113 square opportunities. Thirteen log and seventeen square
opportunities have only one harm-feasible assignment. Sparse eligible pairs,
limited feasible choices and weak coupling all contribute; the solver is not
simply failing to finish.

## Interpretation and Limits

The new pool is not algebraically null everywhere, unlike the preceding fixed
experiment's unchanged optima. However, its observed mechanism support remains
far too concentrated to justify claiming a general multi-agent contribution.
One changed frame is a concrete example for inspecting the current objective,
not evidence that the model learned transferable joint dynamics.

The proximity cost is a fixed geometric proxy based on past-supported rollouts.
It is not a learned interaction loss, real collision measurement or physical
safety certificate. This result does not prove that every joint world model
must fail, or that a future training-supported interaction target is pointless.
It does argue against a broad new pair-weight/threshold search on these same
exposed scenes merely to make this module look useful.

The inherited graph includes 21,897 repeated edges with unsupported context
forecasts, in 5,400 scene/seed queries. Those edges are unpriced, not certified
safe. There are 17,244 unsupported context-row instances. Target forecasts are
past-eligible; future label availability never determines the pool or graph.
No future target or held validity arrays are loaded by this audit. Frozen
train-only preprocessing and scores may of course have been learned from
permitted training labels; that is distinct from outcome inputs at inference.

## Research Decision

Keep the failed loss-matching primary result, conventional forest comparator,
and earlier negative joint controls visible. Do not promote this causal support
audit into another accuracy result or a deployment. Do not tune weights using
the six changed frames. The joint contribution remains unproved.

The next priority is independent scene support and a frozen calibration versus
confirmation role decision, not another generic loss/architecture sweep. The
existing calibration audit shows that reusing heads or removing rows alone is
insufficient: scoring and cost-target producers must also exclude the new
calibration site. The pending independent-source admission decision has not been
silently replaced with another source-internal split.

All 62,796 scene/seed queries and 188,388 pool/query summaries replay exactly.
37 scoped tests pass, including an independent scalar pair-objective check on
synthetic problems and rejection of future arrays. This is same-agent numerical
verification, not independent research replication. New predictive bootstrap is
not applicable because no new prediction-error readout was performed.

Analysis SHA256: `d4165112eab6f47bfc77fdabed6b2054051ccfa01026eaaf6ee844c32b7c92b8`.
See [execution details](execution_notes.md), [analysis](analysis.json) and
[exact replay](replay.json). No deployment change, Stage5C or SMC execution.
No metric/seconds, true-3D, foundation or submission-ready claim is supported.
