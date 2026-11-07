# Optimize The Early Within-Query Easy-Reference Decision

The prior pointwise models improve pooled ranking but miss available
easy reference at the first eligible query. Keep the original observed
easy label, eligible agents, exact causal inputs, and source roles.
Regenerate all fitting-side direct and cost/rank inputs from the other
six fitting sources; no held or calibration source affects weights or
normalization. No new forecaster training.

For each query with both easy and non-easy eligible agents, train the
negative log softmax mass assigned to all easy agents. The two arms
share a 24-unit tanh model, initialization, source/record/query draws,
220 AdamW updates, and 16 queries per update. One weights all mixed
queries equally, the other gives weight eight to queries among the
recording's first two eligible ones (the ordinal counts every eligible
query, even homogeneous ones). Prefix weighting is training only.
Mixed-query training does not remove any query from held scoring.
Fit and normalize on the original seven sources only; there is no
validation selection or early stopping in this component comparison.

Primary: first eligible query top-one hit rate of early-weighted
training versus matched uniform listwise and the two exact frozen
controls (joint easy/safe and pointwise boosted). Report first-two
and all-query rates, and descriptive hits in the 35 missing-support
views. Each model sees the identical eligible agents at each query.
Average recordings within source/seed, seeds within source, and
resample the 14 sources 3,000 times for paired intervals. Advance only
if the early arm has a strictly positive first-query paired 95%
interval over all three comparisons; do not infer policy gain.

Run native arm64 Torch CPU, one process, four computation threads,
zero loader workers. Save optimizer and model every 55 updates,
resume from the exact deterministic draw plan. Freeze scores before
held evaluation. Exposed Moving A 8/12 annotation steps, native
image coordinates, repeated views not independent. No operating
policy, 2% or support-rule change, independent outcome access,
deployment, metric/seconds claim, Stage5C or SMC.
