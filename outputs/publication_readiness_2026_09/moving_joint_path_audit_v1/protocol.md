# Joint Path Audit On The Frozen Native 8/12 Cohort

Use the same exposed 34 recordings, 14 sources and three trained
forecaster seeds. No refitting or policy selection. Evaluate causal
CV and fitting-selected OLS4, and four frozen full-amplitude policies
at equal per-query counts: independent, shared-family scene, additive,
and joint. Separately evaluate the original quarter roster, dense
quarter reranking and both calibrated sparse quarter policies at
their common per-query counts. Do not imply count matching between
these two families or against CV/OLS4.

Freeze the graph to the undirected union of each agent's four nearest
current visible neighbors, using the existing past-only graph and
tie order. Each pair-time is observed only when both agents have
labels. Report observed relative vector ADE and terminal error,
underestimation of observed pair separation, available scene ADE,
the p95 query-level pair error, and worst recording view. These are
image-coordinate diagnostics, not physical collision metrics.

For a paired contrast, subtract right-arm pair error from left-arm
pair error. At an unobserved pair-time, bound the difference by the
norm of the difference of the two predicted relative vectors. Average
these pointwise bounds over the full requested pair grid per query.
Report analogous bounds for the terminal step. This is not a bound
on an absolute missing-label score. Primary: joint vs additive on the
lower bound of the full-grid pair ADE difference.

Average query means within a source and seed, seeds within a source,
then sources equally. Resample sources 3,000 times, not windows.
Require both existing observed pair ADE contrasts to replay. No change
to the original recording-risk protocol and no external outcome access.
