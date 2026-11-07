# Sparse Same-Query Exchange At Quarter Amplitude

Use the existing exposed 34 recordings, 14 source groups and three frozen
forecaster seeds. Start from the exact original quarter-amplitude roster.
Each move replaces one selected agent with a different, currently
unselected agent in that query. Select any legal nonidentity candidate
family for the added agent. Leave every other agent and forecast
unchanged. Allow at most one such exchange per query, and at most 1, 4,
or 16 exchanges per recording. Do not insert an intervention in a
query that had none. All arms retain exactly the same intervention
count per query and quarter amplitude for active forecasts.

With labels, maximize paired completion-lower gain over the original
quarter roster under the unchanged positive all-risk, positive
easy-risk and signed-easy 2% limits, and positive selected easy
reference. Disallow negative paired lower gain. Risk and gains use
the original scorer: per-recording risk masses and query-balanced
paired utility. Unknown-row insertion and removal each pay the actual
forecast disagreement radius.

Cap 1 is exact enumeration; caps 4 and 16 use integer optimization
with a four-second per-view limit. Recheck every witness with the
unchanged scorer. Solver timeouts without valid witnesses remain
unresolved. Report new no-loss repairs and repairs among the original
undefined easy views. The primary comparison is cap 4 versus the
original quarter pass count. This is a capacity diagnosis with
outcomes, not a predictor or trained selector. No witness is a policy
input; no external role is assigned or outcome opened.
