# Multi-Action Allocation On A Fixed Forecast Bank

**Question.** The eight alternatives plus CV cover all original A views for
a label-aware oracle. Can an allocator exploit that coverage using only
past inputs, and does joint allocation help beyond marginal costs?

**Population and roles.** Keep native eight observations and twelve requests,
125,540 rows, 1,613 queries, 34 recordings, 14 source groups. All three
seed-specific banks reuse checked EqMotion, PCT, history-only kinematic and
five fixed causal paths. All three learned producers exclude every A source.
For each actual producer seed, fit the cost model for an A source on the
other 13 A sources only. Source moment sums exclude the target source
directly; normalization is per-row past scale, never held statistics.
A was previously exposed and is development, not external confirmation.

**Costs.** Fit action-conditioned ridge moments with equal fitting-source
weight and equal requested-query mass within source. Fit signed known ADE
benefit, known positive harm, easy harm, selected reference, easy reference
and unknown support. Zero known contribution on wholly unknown rows is an
accounting quantity, not an imputed position. Subtract predicted unknown
probability times exact causal path disagreement from predicted gain.
The model estimates decision costs, not coverage or risk guarantees.

**Matched decisions.** CV-only; independent top-ranked alternatives; one
common candidate family per query; pair-additive; and joint. Four switch
arms choose the same number of distinct changes per query, at most two and
at most n-1, reduced only if no common action can support that count. Rank
row candidates once from predicted gain. The pair variants add the same
marginal costs around the causal OLS4 relative path on four-nearest-peer
edges. Joint adds the exact mixed term for two switches and maximizes over
all pairs of these fixed row candidates. Pair weight .10, ridge .002, no
selection on held results. Secondary variable-count policy uses the largest
positive-utility prefix of joint proposals satisfying both predicted 2%
recording budgets; empty prefix falls back to CV. It is not a proof of
support and is not a same-count contrast.

**Readout.** Freeze choices before scoring. Primary matched comparison is
joint vs pair-additive direct paired completion-lower ADE, with available
relative-pair displacement error as a secondary mechanism check. Also
compare joint vs independent, scene and CV; report exact switch counts,
tail ADE, hard/easy and endpoint availability. Source bootstrap n=3,000
after averaging seeds. Evaluate the unchanged observed recording-level
2% and reference/support rules on all 102 repeated views. Undefined risk
remains undefined, including all-CV outcomes with no selected denominator.
No DroneCrowd access, Stage5C, SMC or change to deployment.
